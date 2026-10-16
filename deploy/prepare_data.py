#!/usr/bin/env python3
"""Checks the wavs, generates split.json (and compares it with the local reference split), fills the oracle cache.

Launched by deploy/prepare_data.sh. Paths read from env.sh. Resumable: the cache skips the tracks already computed.
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
            sys.exit(f"ERROR: {len(dirs)} tracks in {processed}/{split} instead of {n}: rerun deploy/prepare_data.sh")
        for d in dirs:
            frames = {s: sf.info(os.path.join(d, f"{s}.wav")).frames for s in STEMS}  # raises if a wav is missing or truncated
            if len(set(frames.values())) != 1:
                sys.exit(f"ERROR: wavs of different lengths in {d} ({frames}): delete this folder and rerun")
    print(f"wav OK : {n_expected[0]} train + {n_expected[1]} test")


def make_split(cfg):
    dm = MUSDB18DataModule(**cfg.data)
    train, val = dm._load_split(dm._get_track_dirs("train"))
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text())
        got = {"train": [os.path.basename(p) for p in train], "val": [os.path.basename(p) for p in val]}
        if got["train"] != ref["train"] or got["val"] != ref["val"]:
            sys.exit("ERROR: split different from the local split (deploy/split_reference.json): results not comparable. "
                     "Check that the 100 train tracks are identical (manifest) and delete split.json.")
        print(f"split OK: {len(train)} train / {len(val)} val, identical to the local split")
    else:
        print(f"split written: {len(train)} train / {len(val)} val (no local reference to compare with)")


def fill_oracle_cache(cfg, cache_dir, workers, limit=None):
    sr, n_fft, hop = cfg.data.sr, cfg.model.n_fft, cfg.model.hop_length
    processed = cfg.data.dataset_path
    cache = OracleCache(cache_dir, processed, 2, n_fft=n_fft, hop_length=hop)  # same key as src.evaluate (2 channels)
    todo = []
    for d in list_track_dirs(processed, "test"):
        name = os.path.basename(d)
        if cache.get(name, sf.info(os.path.join(d, "mixture.wav")).frames) is None:
            todo.append((name, d))
    todo = todo[:limit] if limit else todo
    print(f"oracle: {len(cache.tracks)} tracks in cache, {len(todo)} to compute ({workers} processes)")
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
            cache.put(futs[fut], part)  # atomic write after each track: resumable
            print(f"  oracle {i}/{len(todo)} : {futs[fut]}", flush=True)
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.update({k: v})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=None, help="processes for the oracle (default: according to cores and RAM, ~3 GB each)")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--limit-oracle", type=int, default=None, help="test: only computes N tracks")
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
    print("data preparation: OK")


if __name__ == "__main__":
    main()
