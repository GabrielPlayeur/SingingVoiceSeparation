#!/usr/bin/env python3
"""Vérifie les wav, génère split.json (et le compare au split local de référence), remplit le cache de l'oracle.

Lancé par deploy/prepare_data.sh. Chemins lus dans env.sh. Reprenable : le cache saute les morceaux déjà calculés.
    python deploy/prepare_data.py [--max-tracks N] [--workers N] [--no-oracle]
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import REPO_ROOT, cpu_cores, load_env, mem_total_gb

sys.path.insert(0, str(REPO_ROOT))
import soundfile as sf  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from src.data import MUSDB18DataModule  # noqa: E402
from src.data.data import list_track_dirs  # noqa: E402
from src.evaluate import score_track  # noqa: E402
from src.models.oracle import OracleCache  # noqa: E402

STEMS = ("mixture", "vocals", "accompaniment")
REFERENCE = Path(__file__).resolve().parent / "split_reference.json"


def check_wavs(processed, n_expected=(100, 50)):
    for split, n in zip(("train", "test"), n_expected):
        dirs = list_track_dirs(processed, split)
        if len(dirs) != n:
            sys.exit(f"ERREUR: {len(dirs)} morceaux dans {processed}/{split} au lieu de {n} : relancer deploy/prepare_data.sh")
        for d in dirs:
            frames = {s: sf.info(os.path.join(d, f"{s}.wav")).frames for s in STEMS}  # lève si un wav manque ou est tronqué
            if len(set(frames.values())) != 1:
                sys.exit(f"ERREUR: wav de longueurs différentes dans {d} ({frames}) : supprimer ce dossier et relancer")
    print(f"wav OK : {n_expected[0]} train + {n_expected[1]} test")


def make_split(cfg):
    dm = MUSDB18DataModule(**cfg.data)
    train, val = dm._load_split(dm._get_track_dirs("train"))
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text())
        got = {"train": [os.path.basename(p) for p in train], "val": [os.path.basename(p) for p in val]}
        if got["train"] != ref["train"] or got["val"] != ref["val"]:
            sys.exit("ERREUR: split différent du split local (deploy/split_reference.json) : résultats non comparables. "
                     "Vérifier que les 100 morceaux train sont identiques (manifeste) et supprimer split.json.")
        print(f"split OK : {len(train)} train / {len(val)} val, identique au split local")
    else:
        print(f"split écrit : {len(train)} train / {len(val)} val (pas de référence locale pour comparer)")


def fill_oracle_cache(cfg, cache_dir, workers, limit=None):
    sr, n_fft, hop = cfg.data.sr, cfg.model.n_fft, cfg.model.hop_length
    processed = cfg.data.dataset_path
    cache = OracleCache(cache_dir, processed, 2, n_fft=n_fft, hop_length=hop)  # même clé que src.evaluate (2 canaux)
    todo = []
    for d in list_track_dirs(processed, "test"):
        name = os.path.basename(d)
        if cache.get(name, sf.info(os.path.join(d, "mixture.wav")).frames) is None:
            todo.append((name, d))
    todo = todo[:limit] if limit else todo
    print(f"oracle : {len(cache.tracks)} morceaux en cache, {len(todo)} à calculer ({workers} processus)")
    if not todo:
        return
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futs = {pool.submit(score_track, dict(kind="oracle", track_dir=d, sr=sr, oracle=dict(n_fft=n_fft, hop_length=hop))): n
                for n, d in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            part = fut.result()
            part.pop("kind")
            cache.put(futs[fut], part)  # écriture atomique après chaque morceau : reprenable
            print(f"  oracle {i}/{len(todo)} : {futs[fut]}", flush=True)
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.update({k: v})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=None, help="processus pour l'oracle (défaut : selon cœurs et RAM, ~3 Go chacun)")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--limit-oracle", type=int, default=None, help="test : ne calcule que N morceaux")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=50)
    args = ap.parse_args()

    env = load_env()
    processed, split_path, cache_dir = env["MUSDB_PROCESSED"], env["SPLIT_PATH"], env["ORACLE_CACHE_DIR"]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "src" / "configs"), version_base=None):
        cfg = compose("config", overrides=[f"data.dataset_path={processed}", f"data.split_path={split_path}", "data.channels=2"])

    check_wavs(processed, (args.n_train, args.n_test))
    make_split(cfg)
    if not args.no_oracle:
        workers = args.workers or max(1, min(cpu_cores(env) - 1, int((mem_total_gb() - 2) // 3)))
        fill_oracle_cache(cfg, cache_dir, workers, args.limit_oracle)
    print("préparation des données : OK")


if __name__ == "__main__":
    main()
