"""
Usage :
    python -m src.evaluate CKPT [--dataset-path musdb18/processed] [--workers 3] [-o scores.json]

Chaque checkpoint est évalué dans deux protocoles, mono et stéréo (voir `evaluate`), pour comparer à conditions égales un
modèle mono et un modèle stéréo.

L'inférence tourne sur le GPU dans le processus principal ; le calcul des métriques (museval, oracle, SI-SDRi), qui est sur
CPU et domine le temps, est confié à un pool de processus. Les scores de l'oracle ne dépendent pas du modèle : ils sont mis
en cache dans `outputs/oracle_cache/` (un fichier par configuration canaux / n_fft / hop).
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


def median_sdr(vocals_true, accomp_true, vocals_est, accomp_est, sr):
    # (C, T) -> (sources, samples, canaux), un seul tableau par côté (pas de copie intermédiaire : la RAM croît avec la durée)
    def stack(a, b):
        out = np.empty((2, a.shape[1], a.shape[0]), dtype=np.float32)
        out[0], out[1] = a.T, b.T
        return out
    sdr, isr, sir, sar = bsseval(stack(vocals_true, accomp_true), stack(vocals_est, accomp_est), win=sr, hop=sr) # 1sec, museval v4 à mémoire bornée (src/utils/fast_museval.py)
    return float(np.nanmedian(sdr[0]))


def si_sdri(vocals_est_t, mixture_t, vocals_true_t):
    return float(si_sdr(vocals_est_t, vocals_true_t) - si_sdr(mixture_t, vocals_true_t))


def default_workers(per_worker_gb=2.5, reserve_gb=2.0, max_workers=3):
    try:
        with open("/proc/meminfo") as f:
            available_gb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 1e6
    except (OSError, StopIteration):
        return 1
    return int(max(1, min(max_workers, (os.cpu_count() or 2) - 1, (available_gb - reserve_gb) // per_worker_gb)))


PROTOCOLS = {"mono": 1, "stereo": 2}  # nom du protocole -> nombre de canaux des références et de l'estimation


def score_track(task):
    """Métriques d'un morceau dans un protocole (CPU, exécuté dans un processus du pool).

    `task["protocol"]` fixe le nombre de canaux des références relues sur disque ("mono" : moyenne L/R, "stereo" : L et R).
    `task["kind"]` vaut "model" (estimation lue dans un .npy, supprimé après lecture) ou "oracle" (masque idéal).
    """
    vocals_est = None
    if task["kind"] == "model":
        try:
            vocals_est = np.load(task["est_path"])
        finally:
            os.remove(task["est_path"])
    sample = MUSDBFullTrackDataset(track_dirs=[task["track_dir"]], sr=task["sr"], channels=PROTOCOLS[task["protocol"]])[0]
    mixture_t, vocals_true_t = sample["mixture"], sample["vocals"]
    mixture, vocals_true = mixture_t.numpy(), vocals_true_t.numpy()
    if task["kind"] == "oracle":
        vocals_est = Oracle(**task["oracle"]).evaluate(mixture, vocals_true)
    return {
        "kind": task["kind"],
        "protocol": task["protocol"],
        "n_samples": int(mixture.shape[-1]),
        "sdr": median_sdr(vocals_true, mixture - vocals_true, vocals_est, mixture - vocals_est, task["sr"]),
        "si_sdri": si_sdri(torch.from_numpy(vocals_est), mixture_t, vocals_true_t),
    }


def evaluate(checkpoint_path, dataset_path, crop_length=261632, batch_size=16, max_tracks=None, logger=None, output_json=None,
             with_oracle=True, workers=None, oracle_cache_dir="outputs/oracle_cache"):
    """Évalue un checkpoint sur le jeu de test dans LES DEUX protocoles, quel que soit le nombre de canaux du modèle, pour
    que deux modèles soient toujours comparables à protocole égal :

    - mono   : références et estimation mixées en mono (moyenne L/R). Modèle stéréo : moyenne de ses deux sorties.
    - stereo : références et estimation sur 2 canaux. Modèle mono : appliqué séparément à L et à R.

    Clés par morceau : `sdr_mono`, `si_sdri_mono`, `sdr_stereo`, `si_sdri_stereo` (+ `*_oracle_*`). Les clés sans suffixe
    (`sdr`, `si_sdri`, `sdr_oracle`, `si_sdri_oracle`) sont celles du protocole natif du modèle (son nombre de canaux).

    `workers` : processus de calcul des métriques (0 = dans le processus principal, None = selon la RAM disponible, voir
    `default_workers`). Chacun a besoin d'environ 2,5 Go, davantage pour un morceau de plus de 6 min."""
    workers = default_workers() if workers is None else workers
    module = VoiceSeparationModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    sr = module.hparams.sr
    channels = module.hparams.in_channels
    native = next(name for name, c in PROTOCOLS.items() if c == channels)
    assert_no_temporal_padding(crop_length, module.hparams.hop_length, module.hparams.depth)
    trainer = Trainer( accelerator="auto",
                       devices=1,
                       logger=False,
                       enable_checkpointing=False,
                       enable_progress_bar=False
                    )
    oracle_cfg = dict(n_fft=module.hparams.n_fft, hop_length=module.hparams.hop_length)
    caches = {name: OracleCache(oracle_cache_dir, dataset_path, c, **oracle_cfg) for name, c in PROTOCOLS.items()} if with_oracle else {}

    track_dirs = list_track_dirs(dataset_path, "test")
    datasets = {name: MUSDBFullTrackDataset(track_dirs=track_dirs, sr=sr, channels=c) for name, c in PROTOCOLS.items()}

    n_tracks = len(track_dirs)
    if max_tracks is not None:
        n_tracks = min(n_tracks, max_tracks)
    results = [{"track": os.path.basename(d)} for d in track_dirs[:n_tracks]]
    kinds = ("", "_oracle") if with_oracle else ("",)
    required = [f"{metric}{kind}_{name}" for name in PROTOCOLS for kind in kinds for metric in ("sdr", "si_sdri")]
    completed = set()
    bar = tqdm(total=n_tracks, desc="Évaluation", unit="morceau")

    def collect(index, part):
        r = results[index]
        kind = "_oracle" if part["kind"] == "oracle" else ""
        scores = {f"{metric}{kind}_{part['protocol']}": part[metric] for metric in ("sdr", "si_sdri")}
        r.update(scores)
        if part["kind"] == "oracle":
            caches[part["protocol"]].put(r["track"], {"sdr_oracle": part["sdr"], "si_sdri_oracle": part["si_sdri"],
                                                      "n_samples": part["n_samples"]})
        if index not in completed and all(k in r for k in required):
            completed.add(index)
            bar.update(1)
            bar.set_postfix({f"SDR_{name}": f"{np.nanmedian([results[i][f'sdr_{name}'] for i in completed]):.2f}" for name in PROTOCOLS})

    pool = None
    pending = {}  # future -> index
    # un thread BLAS par processus : sinon `workers` processus se disputent tous les cœurs. Les processus du pool sont créés à la
    # demande et héritent de l'environnement du parent : les variables restent donc posées jusqu'à la fin de l'évaluation.
    saved_env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if workers > 0:
        os.environ.update({k: "1" for k in saved_env})
        pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))

    def submit(index, task):
        if pool is None:
            collect(index, score_track(task))
            return
        while len(pending) >= workers + 1:  # borne le nombre d'estimations en attente sur disque
            finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                collect(pending.pop(fut), fut.result())
        pending[pool.submit(score_track, task)] = index

    def separate(mixture):
        return separate_full_track(module, mixture, crop_length, batch_size=batch_size, trainer=trainer)

    try:
        with tempfile.TemporaryDirectory(prefix="eval_est_") as tmp:
            for i in range(n_tracks):
                base = dict(track_dir=track_dirs[i], sr=sr)
                stereo_mix = datasets["stereo"][i]["mixture"]  # (2, T)
                n_samples = int(stereo_mix.shape[-1])
                for name in caches:
                    cached = caches[name].get(results[i]["track"], n_samples)
                    if cached is None:
                        submit(i, dict(base, kind="oracle", protocol=name, oracle=oracle_cfg))
                    else:
                        results[i].update({f"sdr_oracle_{name}": cached["sdr_oracle"], f"si_sdri_oracle_{name}": cached["si_sdri_oracle"]})

                if channels == 2:
                    stereo_est = separate(stereo_mix)
                    estimates = {"stereo": stereo_est, "mono": stereo_est.mean(dim=0, keepdim=True)}
                else:
                    # le mono natif voit le même signal qu'à l'entraînement ; en stéréo, le modèle traite L puis R
                    estimates = {"mono": separate(datasets["mono"][i]["mixture"]),
                                 "stereo": torch.cat([separate(stereo_mix[c:c + 1]) for c in range(2)])}
                for name, est in estimates.items():
                    est_path = os.path.join(tmp, f"{i}_{name}.npy")
                    np.save(est_path, est.numpy())
                    submit(i, dict(base, kind="model", protocol=name, est_path=est_path))
            for fut in cf.as_completed(list(pending)):
                collect(pending.pop(fut), fut.result())
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for k, v in saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
        bar.close()

    # clés sans suffixe = protocole natif du modèle (compatibilité avec les JSON et les scripts d'analyse existants)
    native_keys = [f"{metric}{kind}" for kind in kinds for metric in ("sdr", "si_sdri")]
    results = [{"track": r["track"], **{k: r[f"{k}_{native}"] for k in native_keys}, **{k: r[k] for k in required}} for r in results]

    print(f"\n{len(results)} morceaux, protocole natif du modèle : {native}. Médiane [IC95% bootstrap] en dB")
    summary = {}
    for name in PROTOCOLS:
        for kind, label in zip(kinds, ("modèle", "oracle")):
            line = []
            for metric in ("sdr", "si_sdri"):
                values = np.array([r[f"{metric}{kind}_{name}"] for r in results])
                med, lo, hi = bootstrap_ci(values)
                line.append(f"{'SDR' if metric == 'sdr' else 'SI-SDRi'} {med:5.2f} [{lo:.2f}, {hi:.2f}]")
                tags = [f"test/{metric}{kind}_{name}"] + ([f"test/{metric}{kind}"] if name == native else [])
                for tag in tags:
                    summary.update({f"{tag}_median": med, f"{tag}_mean": float(np.nanmean(values)),
                                    f"{tag}_ci_low": lo, f"{tag}_ci_high": hi})
            print(f"  {name:6s} {label:6s} : " + "   ".join(line))

    if output_json is not None:
        # scores par morceau, pour comparer deux modèles par test apparié (même morceaux)
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
    parser.add_argument('-o', "--output-json", default=None, help="écrit les scores par morceau")
    parser.add_argument("--no-oracle", action="store_true", help="n'évalue pas l'oracle (borne haute)")
    parser.add_argument('-w', "--workers", type=int, default=None,
                        help="processus de calcul des métriques (0 = séquentiel ; défaut : selon la RAM disponible, ~2,5 Go chacun)")
    parser.add_argument("--oracle-cache", default="outputs/oracle_cache", help="dossier du cache de l'oracle ('' pour le désactiver)")
    args = parser.parse_args()
    evaluate(args.ckpt, args.dataset_path, max_tracks=args.max_tracks, output_json=args.output_json,
             with_oracle=not args.no_oracle, workers=args.workers, oracle_cache_dir=args.oracle_cache or None)
