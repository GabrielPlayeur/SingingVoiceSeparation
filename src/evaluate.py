"""
Usage :
    python -m src.evaluate CKPT [--dataset-path musdb18/processed]
"""
import argparse
import json

import numpy as np
import museval
import torch
from lightning import Trainer
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from .data.data import list_track_dirs
from .data.dataset import MUSDBFullTrackDataset
from .module import VoiceSeparationModule
from .inference import separate_full_track
from .utils import assert_no_temporal_padding


def bootstrap_ci(values, n_resamples=10_000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    medians = np.median(values[idx], axis=1)
    lo, hi = np.quantile(medians, [alpha / 2, 1 - alpha / 2])
    return float(np.median(values)), float(lo), float(hi)


def evaluate(checkpoint_path, dataset_path, crop_length=261632, batch_size=16, max_tracks=None, logger=None, output_json=None):
    module = VoiceSeparationModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    sr = module.hparams.sr
    assert_no_temporal_padding(crop_length, module.hparams.hop_length, module.hparams.depth)
    trainer = Trainer( accelerator="auto",
                       devices=1,
                       logger=False,
                       enable_checkpointing=False,
                       enable_progress_bar=False
                    )

    test_dataset = MUSDBFullTrackDataset(track_dirs=list_track_dirs(dataset_path, "test"), sr=sr)

    n_tracks = len(test_dataset)
    if max_tracks is not None:
        n_tracks = min(n_tracks, max_tracks)
    results = []

    for i in range(n_tracks):
        sample = test_dataset[i]
        mixture = sample["mixture"]
        vocals_true_t = sample["vocals"]
        vocals_true = vocals_true_t.numpy()
        accomp_true = (mixture - vocals_true_t).numpy()

        vocals_est_t = separate_full_track(module, mixture, crop_length, batch_size=batch_size, trainer=trainer)
        vocals_est = vocals_est_t.numpy()
        accomp_est = mixture.numpy() - vocals_est

        # protocole SigSep : on évalue voix ET accompagnement ensemble (sources, temps, canaux)
        references = np.stack([vocals_true, accomp_true])[..., np.newaxis]
        estimates = np.stack([vocals_est, accomp_est])[..., np.newaxis]
        sdr, isr, sir, sar = museval.evaluate(references, estimates, win=sr, hop=sr) # 1sec

        median_sdr = float(np.nanmedian(sdr[0]))

        # gain SI-SDR vs mix brut : indépendant de l'énergie du morceau, contrairement au SI-SDR brut
        si_sdri = float(
            scale_invariant_signal_distortion_ratio(vocals_est_t, vocals_true_t)
            - scale_invariant_signal_distortion_ratio(mixture, vocals_true_t)
        )

        results.append({"track": sample["track_name"], "sdr": median_sdr, "si_sdri": si_sdri})
        print(f"{sample['track_name']}: SDR = {median_sdr:.2f} dB, SI-SDRi = {si_sdri:.2f} dB")

    scores = np.array([r["sdr"] for r in results])
    si_sdri_scores = np.array([r["si_sdri"] for r in results])
    sdr_med, sdr_lo, sdr_hi = bootstrap_ci(scores)
    sisdri_med, sisdri_lo, sisdri_hi = bootstrap_ci(si_sdri_scores)
    print(f"\nSDR médian sur {len(scores)} morceaux : {sdr_med:.2f} dB  IC95% bootstrap [{sdr_lo:.2f}, {sdr_hi:.2f}]")
    print(f"SI-SDRi médian sur {len(si_sdri_scores)} morceaux : {sisdri_med:.2f} dB  IC95% bootstrap [{sisdri_lo:.2f}, {sisdri_hi:.2f}]")

    if output_json is not None:
        # scores par morceau, pour comparer deux modèles par test apparié (même morceaux)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    if logger is not None:
        import wandb
        run = logger.experiment
        table = wandb.Table(
            columns=["track", "sdr", "si_sdri"],
            data=[[r["track"], r["sdr"], r["si_sdri"]] for r in results],
        )
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
        run.log({**summary, "test/per_track": table})
        run.summary.update(summary)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--dataset-path", default="musdb18/processed")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--output-json", default=None, help="écrit les scores par morceau")
    args = parser.parse_args()
    evaluate(args.ckpt, args.dataset_path, max_tracks=args.max_tracks, output_json=args.output_json)
