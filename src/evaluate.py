"""
Usage:
    python -m src.evaluate CKPT [--dataset-path musdb18/processed] [--workers 3] [-o scores.json]

Evaluation always in stereo (2 channels), the protocol of the literature: see `evaluate`.

Inference runs on the GPU in the main process; the computation of the metrics (museval, oracle, SI-SDRi), which is on the
CPU and dominates the time, is handed to a pool of processes. The oracle scores do not depend on the model: they are cached
in `outputs/oracle_cache/` (one file per channels / n_fft / hop configuration, plus one per mixture convention).
"""
import argparse
import concurrent.futures as cf
import json
import logging
import multiprocessing
import os
import tempfile
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

import numpy as np
import torch
from lightning import Trainer
from tqdm import tqdm

from .data.data import list_track_dirs
from .data.dataset import MUSDBFullTrackDataset
from .module import VoiceSeparationModule
from .inference import separate_full_track
from .utils.utils import assert_no_temporal_padding, si_sdr
from .models.oracle import Oracle, OracleCache
from .utils.fast_museval import evaluate as bsseval


def bootstrap_ci(values, n_resamples=10_000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    medians = np.median(values[idx], axis=1)
    lo, hi = np.quantile(medians, [alpha / 2, 1 - alpha / 2])
    return float(np.median(values)), float(lo), float(hi)


def bss_medians(vocals_true, accomp_true, vocals_est, accomp_est, sr):
    """Median over the 1 s windows of the vocals' SDR, SIR and SAR (museval v4, bounded memory: src/utils/fast_museval.py).
    SIR measures the leakage of the accompaniment into the estimate, SAR the artifacts (Vincent et al. 2006, BSS Eval)."""
    # (C, T) -> (sources, samples, channels), a single array per side (no intermediate copy: RAM grows with the duration)
    def stack(a, b):
        out = np.empty((2, a.shape[1], a.shape[0]), dtype=np.float32)
        out[0], out[1] = a.T, b.T
        return out
    sdr, isr, sir, sar = bsseval(stack(vocals_true, accomp_true), stack(vocals_est, accomp_est), win=sr, hop=sr) # 1sec
    return {"sdr": float(np.nanmedian(sdr[0])), "sir": float(np.nanmedian(sir[0])), "sar": float(np.nanmedian(sar[0]))}


def median_sdr(vocals_true, accomp_true, vocals_est, accomp_est, sr):
    return bss_medians(vocals_true, accomp_true, vocals_est, accomp_est, sr)["sdr"]


def si_sdri(vocals_est_t, mixture_t, vocals_true_t):
    return float(si_sdr(vocals_est_t, vocals_true_t) - si_sdr(mixture_t, vocals_true_t))


def default_workers(per_worker_gb=2.5, reserve_gb=2.0, max_workers=3):
    try:
        with open("/proc/meminfo") as f:
            available_gb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 1e6
    except (OSError, StopIteration):
        return 1
    return int(max(1, min(max_workers, (os.cpu_count() or 2) - 1, (available_gb - reserve_gb) // per_worker_gb)))


def score_track(task):
    """Metrics of one track in stereo (2 channels), on CPU, executed in a pool process.

    `task["kind"]` is "model" (estimate read from a .npy, deleted after reading) or "oracle" (ideal mask).
    """
    vocals_est = None
    if task["kind"] == "model":
        try:
            vocals_est = np.load(task["est_path"])
        finally:
            os.remove(task["est_path"])
    official = task.get("mixture", "sum") == "official"
    sample = MUSDBFullTrackDataset(track_dirs=[task["track_dir"]], sr=task["sr"], channels=2, mixture=task.get("mixture", "sum"))[0]
    mixture_t, vocals_true_t = sample["mixture"], sample["vocals"]
    mixture, vocals_true = mixture_t.numpy(), vocals_true_t.numpy()
    if task["kind"] == "oracle":
        vocals_est = Oracle(**task["oracle"]).evaluate(mixture, vocals_true)
    # accompaniment reference: the sum convention defines it as mixture - vocals (kept so that the "sum" scores do not change);
    # with the official mixture, the reference is the accompaniment stem itself and the estimate is mixture - vocals_est
    accomp_true = sample["accompaniment"].numpy() if official else mixture - vocals_true
    suffix = "_oracle" if task["kind"] == "oracle" else ""
    scores = bss_medians(vocals_true, accomp_true, vocals_est, mixture - vocals_est, task["sr"])
    return {
        "kind": task["kind"],
        "n_samples": int(mixture.shape[-1]),
        **{k + suffix: v for k, v in scores.items()},
        "si_sdri" + suffix: si_sdri(torch.from_numpy(vocals_est), mixture_t, vocals_true_t),
    }


def evaluate(checkpoint_path, dataset_path, crop_length=261632, batch_size=16, max_tracks=None, logger=None, output_json=None,
             with_oracle=True, workers=None, oracle_cache_dir="outputs/oracle_cache", module=None, accelerator="auto",
             mixture="sum", separate_fn=None, sr=44100):
    """Evaluates a checkpoint on the test set, always in STEREO: references and estimate on 2 channels (museval v4 SDR
    on 2 channels), the protocol of the literature. A mono model (old checkpoint) is applied separately to L then to R.

    `mixture`: "sum" (default; mixture = vocals + accompaniment, as in training) or "official" (mixture.wav, the MUSDB18
    mixture: protocol of the literature, see `MUSDBFullTrackDataset`). The model and the oracle receive this mixture, and
    the accompaniment estimate is mixture - estimated vocals.

    `workers`: processes computing the metrics (0 = in the main process, None = according to the available RAM, see
    `default_workers`). Each needs about 2.5 GB, more for a track longer than 6 min.

    `separate_fn`: external model (e.g. Open-Unmix, src/evaluate_umx.py), function (2, T) -> (2, T) returning the estimated vocals at
    `sr`; it then replaces the checkpoint, and the oracle is disabled (it is tied to the U-Net's STFT)."""
    workers = default_workers() if workers is None else workers
    if separate_fn is None:
        if module is None:  # `module`: model already built (e.g. quantized U-Net, src/quantize.py); checkpoint_path then serves as a label
            module = VoiceSeparationModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
        sr = module.hparams.sr
        channels = module.hparams.in_channels
        assert_no_temporal_padding(crop_length, module.hparams.hop_length, module.hparams.depth)
        trainer = Trainer( accelerator=accelerator,
                           devices=1,
                           logger=False,
                           enable_checkpointing=False,
                           enable_progress_bar=False
                        )
        oracle_cfg = dict(n_fft=module.hparams.n_fft, hop_length=module.hparams.hop_length)
    else:
        channels, with_oracle = 2, False
    cache = OracleCache(oracle_cache_dir, dataset_path, 2, mixture=mixture, **oracle_cfg) if with_oracle else None

    track_dirs = list_track_dirs(dataset_path, "test")
    test_dataset = MUSDBFullTrackDataset(track_dirs=track_dirs, sr=sr, channels=2, mixture=mixture)

    n_tracks = len(track_dirs)
    if max_tracks is not None:
        n_tracks = min(n_tracks, max_tracks)
    results = [{"track": os.path.basename(d)} for d in track_dirs[:n_tracks]]
    metrics = ("sdr", "sir", "sar", "si_sdri")
    required = list(metrics) + ([m + "_oracle" for m in metrics] if with_oracle else [])
    completed = set()
    bar = tqdm(total=n_tracks, desc="Evaluation", unit="track")

    def collect(index, part):
        r = results[index]
        kind, n_samples = part.pop("kind"), part.pop("n_samples")
        r.update(part)
        if kind == "oracle":
            cache.put(r["track"], {**part, "n_samples": n_samples})
        if index not in completed and all(k in r for k in required):
            completed.add(index)
            bar.update(1)
            bar.set_postfix(median_SDR=f"{np.nanmedian([results[i]['sdr'] for i in completed]):.2f} dB")

    pool = None
    pending = {}  # future -> index
    # one BLAS thread per process: otherwise `workers` processes fight over all the cores. The pool processes are created on
    # demand and inherit the parent's environment: the variables therefore stay set until the end of the evaluation.
    saved_env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if workers > 0:
        os.environ.update({k: "1" for k in saved_env})
        pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))

    def submit(index, task):
        if pool is None:
            collect(index, score_track(task))
            return
        while len(pending) >= workers + 1:  # bounds the number of estimates waiting on disk
            finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                collect(pending.pop(fut), fut.result())
        pending[pool.submit(score_track, task)] = index

    try:
        with tempfile.TemporaryDirectory(prefix="eval_est_") as tmp:
            for i in range(n_tracks):
                stereo_mix = test_dataset[i]["mixture"]  # (2, T)
                base = dict(track_dir=track_dirs[i], sr=sr, mixture=mixture)
                if with_oracle:
                    cached = cache.get(results[i]["track"], int(stereo_mix.shape[-1]))
                    if cached is None:
                        submit(i, dict(base, kind="oracle", oracle=oracle_cfg))
                    else:
                        results[i].update({k: v for k, v in cached.items() if k != "n_samples"})
                if separate_fn is not None:
                    est = separate_fn(stereo_mix)
                elif channels == 2:
                    est = separate_full_track(module, stereo_mix, crop_length, batch_size=batch_size, trainer=trainer)
                else:  # old mono model: applied to L then to R
                    est = torch.cat([separate_full_track(module, stereo_mix[c:c + 1], crop_length, batch_size=batch_size,
                                                         trainer=trainer) for c in range(2)])
                est_path = os.path.join(tmp, f"{i}.npy")
                np.save(est_path, est.numpy())
                submit(i, dict(base, kind="model", est_path=est_path))
            for fut in cf.as_completed(list(pending)):
                collect(pending.pop(fut), fut.result())
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for k, v in saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
        bar.close()

    order = ("track", *metrics, *(m + "_oracle" for m in metrics))
    results = [{k: r[k] for k in order if k in r} for r in results]

    print(f"\n{len(results)} tracks, stereo protocol, mixture {mixture}. Median [95% bootstrap CI] in dB")
    summary = {}
    for kind, label in (("", "model"), ("_oracle", "oracle")):
        if kind and not with_oracle:
            continue
        line = []
        for metric, name in (("sdr", "SDR"), ("sir", "SIR"), ("sar", "SAR"), ("si_sdri", "SI-SDRi")):
            values = np.array([r[f"{metric}{kind}"] for r in results])
            med, lo, hi = bootstrap_ci(values)
            line.append(f"{name} {med:5.2f} [{lo:.2f}, {hi:.2f}]")
            summary.update({f"test/{metric}{kind}_median": med, f"test/{metric}{kind}_mean": float(np.nanmean(values)),
                            f"test/{metric}{kind}_ci_low": lo, f"test/{metric}{kind}_ci_high": hi})
        print(f"  {label:6s} : " + "   ".join(line))

    if output_json is not None:
        # per-track scores, to compare two models with a paired test (same tracks)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    if logger is not None:
        import wandb
        run = logger.experiment
        columns = list(results[0])
        table = wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in results])
        run.log({**summary, "test/per_track": table})
        run.summary.update(summary)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--dataset-path", default="musdb18/processed")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument('-o', "--output-json", default=None, help="writes the per-track scores")
    parser.add_argument("--no-oracle", action="store_true", help="does not evaluate the oracle (upper bound)")
    parser.add_argument('-w', "--workers", type=int, default=None,
                        help="processes computing the metrics (0 = sequential; default: according to the available RAM, ~2.5 GB each)")
    parser.add_argument("--oracle-cache", default="outputs/oracle_cache", help="folder of the oracle cache ('' to disable it)")
    parser.add_argument("--mixture", choices=("sum", "official"), default="sum",
                        help="mixture given to the model: 'sum' = vocals + accompaniment (default, training convention), 'official' = "
                             "mixture.wav of MUSDB18 (protocol of the literature)")
    parser.add_argument("--crop-length", type=int, default=261632,
                        help="length (in samples) of the inference windows, multiple of 2**depth STFT frames; default: the one of the baseline training "
                             "(5.93 s). Used by the context-length experiment.")
    args = parser.parse_args()
    evaluate(args.ckpt, args.dataset_path, crop_length=args.crop_length, max_tracks=args.max_tracks, output_json=args.output_json,
             with_oracle=not args.no_oracle, workers=args.workers, oracle_cache_dir=args.oracle_cache or None, mixture=args.mixture)
