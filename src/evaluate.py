"""
Usage :
    python -m src.evaluate CKPT [--dataset-path musdb18/processed]
"""
import argparse
import json
import logging
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

import numpy as np
import museval
import torch
from lightning import Trainer
from tqdm import tqdm

from .data.data import list_track_dirs
from .data.dataset import MUSDBFullTrackDataset
from .module import VoiceSeparationModule
from .inference import separate_full_track
from .utils import assert_no_temporal_padding, si_sdr
from .oracle import Oracle


def bootstrap_ci(values, n_resamples=10_000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    medians = np.median(values[idx], axis=1)
    lo, hi = np.quantile(medians, [alpha / 2, 1 - alpha / 2])
    return float(np.median(values)), float(lo), float(hi)


def median_sdr(vocals_true, accomp_true, vocals_est, accomp_est, sr):
    # (C, T) -> (sources, samples, canaux)
    references = np.stack([vocals_true, accomp_true]).transpose(0, 2, 1)
    estimates = np.stack([vocals_est, accomp_est]).transpose(0, 2, 1)
    sdr, isr, sir, sar = museval.evaluate(references, estimates, win=sr, hop=sr) # 1sec
    return float(np.nanmedian(sdr[0]))


def si_sdri(vocals_est_t, mixture_t, vocals_true_t):
    return float(si_sdr(vocals_est_t, vocals_true_t) - si_sdr(mixture_t, vocals_true_t))


def evaluate(checkpoint_path, dataset_path, crop_length=261632, batch_size=16, max_tracks=None, logger=None, output_json=None, with_oracle=True):
    module = VoiceSeparationModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    sr = module.hparams.sr
    assert_no_temporal_padding(crop_length, module.hparams.hop_length, module.hparams.depth)
    trainer = Trainer( accelerator="auto",
                       devices=1,
                       logger=False,
                       enable_checkpointing=False,
                       enable_progress_bar=False
                    )
    oracle = Oracle( n_fft=module.hparams.n_fft,
                     hop_length=module.hparams.hop_length
                    ) if with_oracle else None

    test_dataset = MUSDBFullTrackDataset(track_dirs=list_track_dirs(dataset_path, "test"), sr=sr, channels=module.hparams.in_channels)

    n_tracks = len(test_dataset)
    if max_tracks is not None:
        n_tracks = min(n_tracks, max_tracks)
    results = []

    bar = tqdm(range(n_tracks), desc="Évaluation", unit="morceau")
    for i in bar:
        sample = test_dataset[i]
        mixture_t = sample["mixture"]
        mixture = mixture_t.numpy()
        vocals_true_t = sample["vocals"]
        vocals_true = vocals_true_t.numpy()
        accomp_true = (mixture_t - vocals_true_t).numpy()

        vocals_est_t = separate_full_track(module, mixture_t, crop_length, batch_size=batch_size, trainer=trainer)
        vocals_est = vocals_est_t.numpy()
        accomp_est = mixture - vocals_est

        result = {
            "track": sample["track_name"],
            "sdr": median_sdr(vocals_true, accomp_true, vocals_est, accomp_est, sr),
            "si_sdri": si_sdri(vocals_est_t, mixture_t, vocals_true_t),
        }

        if oracle is not None:
            vocals_ora = oracle.evaluate(mixture, vocals_true)
            result["sdr_oracle"] = median_sdr(vocals_true, accomp_true, vocals_ora, mixture - vocals_ora, sr)
            result["si_sdri_oracle"] = si_sdri(torch.from_numpy(vocals_ora), mixture_t, vocals_true_t)

        results.append(result)
        bar.set_postfix(SDR_médian=f"{np.nanmedian([r['sdr'] for r in results]):.2f} dB")

    scores = np.array([r["sdr"] for r in results])
    si_sdri_scores = np.array([r["si_sdri"] for r in results])
    sdr_med, sdr_lo, sdr_hi = bootstrap_ci(scores)
    sisdri_med, sisdri_lo, sisdri_hi = bootstrap_ci(si_sdri_scores)
    print(f"\nSDR médian sur {len(scores)} morceaux : {sdr_med:.2f} dB  IC95% bootstrap [{sdr_lo:.2f}, {sdr_hi:.2f}]")
    print(f"SI-SDRi médian sur {len(si_sdri_scores)} morceaux : {sisdri_med:.2f} dB  IC95% bootstrap [{sisdri_lo:.2f}, {sisdri_hi:.2f}]")

    if oracle is not None:
        sdr_ora_scores = np.array([r["sdr_oracle"] for r in results])
        si_sdri_ora_scores = np.array([r["si_sdri_oracle"] for r in results])
        sdr_ora_med, sdr_ora_lo, sdr_ora_hi = bootstrap_ci(sdr_ora_scores)
        sisdri_ora_med, sisdri_ora_lo, sisdri_ora_hi = bootstrap_ci(si_sdri_ora_scores)
        print(f"SDR médian oracle : {sdr_ora_med:.2f} dB  IC95% bootstrap [{sdr_ora_lo:.2f}, {sdr_ora_hi:.2f}]")
        print(f"SI-SDRi médian oracle : {sisdri_ora_med:.2f} dB  IC95% bootstrap [{sisdri_ora_lo:.2f}, {sisdri_ora_hi:.2f}]")

    if output_json is not None:
        # scores par morceau, pour comparer deux modèles par test apparié (même morceaux)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    if logger is not None:
        import wandb
        run = logger.experiment
        columns = ["track", "sdr", "si_sdri"] + (["sdr_oracle", "si_sdri_oracle"] if oracle is not None else [])
        table = wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in results])
        summary = {
            "test/sdr_median": float(np.nanmedian(scores)),
            "test/sdr_mean": float(np.nanmean(scores)),
            "test/si_sdri_median": float(np.nanmedian(si_sdri_scores)),
            "test/si_sdri_mean": float(np.nanmean(si_sdri_scores)),
            "test/sdr_ci_low": sdr_lo,
            "test/sdr_ci_high": sdr_hi,
            "test/si_sdri_ci_low": sisdri_lo,
            "test/si_sdri_ci_high": sisdri_hi,
        }
        if oracle is not None:
            summary.update({
                "test/sdr_oracle_median": float(np.nanmedian(sdr_ora_scores)),
                "test/sdr_oracle_mean": float(np.nanmean(sdr_ora_scores)),
                "test/si_sdri_oracle_median": float(np.nanmedian(si_sdri_ora_scores)),
                "test/si_sdri_oracle_mean": float(np.nanmean(si_sdri_ora_scores)),
                "test/sdr_oracle_ci_low": sdr_ora_lo,
                "test/sdr_oracle_ci_high": sdr_ora_hi,
                "test/si_sdri_oracle_ci_low": sisdri_ora_lo,
                "test/si_sdri_oracle_ci_high": sisdri_ora_hi,
            })
        run.log({**summary, "test/per_track": table})
        run.summary.update(summary)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--dataset-path", default="musdb18/processed")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument('-o', "--output-json", default=None, help="écrit les scores par morceau")
    parser.add_argument("--no-oracle", action="store_true", help="n'évalue pas l'oracle (borne haute)")
    args = parser.parse_args()
    evaluate(args.ckpt, args.dataset_path, max_tracks=args.max_tracks, output_json=args.output_json, with_oracle=not args.no_oracle)
