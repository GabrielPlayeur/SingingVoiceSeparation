#!/usr/bin/env python3
"""Evaluates already-trained checkpoints at OTHER inference window lengths ("context length" experiment,
experiments/07_context_length.md). Nothing is retrained: `src.evaluate --crop-length N` on the checkpoint of the run chosen by its queue entry (`ckpt`, default best).

    python deploy/eval_crops.py --tag ctx-v1 --eval ctx3_s42=130560 --eval ctx12_s42=523776 \
        --eval loss-v1/baseline_s42=130560,392704,523776 [--parallel 2] [--dry-run]

- `--eval RUN=N[,N...]`: RUN is a folder of RUNS_DIR/<tag>/; `OTHER_TAG/RUN` designates a run of another tag (e.g. the baseline of
  loss-v1): its results go to RUNS_DIR/<tag>/infer_<OTHER_TAG>_<RUN>/ so that they leave with the fetch of <tag>.
- Output: eval_crop<N>.json (+ .log) next to eval.json, same format. Already present: skipped (--force to redo).
- N is in samples and must give a number of STFT frames that is a multiple of 2**depth: 130560 (2.96 s), 261632 (5.93 s),
  392704 (8.90 s), 523776 (11.88 s).
- The code that runs is the frozen checkout of <tag> (RUNS_DIR/<tag>/_code, created by run_queue.py): it must contain `--crop-length`.
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import REPO_ROOT, cpu_cores, find_ckpt, load_env


def parse(spec, tag, runs_dir):
    run, _, crops = spec.partition("=")
    if not crops:
        sys.exit(f"ERROR: --eval {spec!r}: expected RUN=N[,N...]")
    if "/" in run:
        other, name = run.split("/", 1)
        src, out = runs_dir / other / name, runs_dir / tag / f"infer_{other}_{name}"
    else:
        src = out = runs_dir / tag / run
    return src, out, [int(c) for c in crops.split(",")]


def main():
    env = load_env()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval", action="append", required=True, metavar="RUN=N[,N...]")
    ap.add_argument("--parallel", type=int, default=1, help="simultaneous evaluations (default 1)")
    ap.add_argument("--workers", type=int, default=None, help="metric processes per evaluation (default: cores / (2 x parallel), at most 6)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(env["RUNS_DIR"])
    code = runs_dir / args.tag / "_code"
    try:
        cwd = code if code.exists() else REPO_ROOT
    except OSError:   # folder not accessible (e.g. --dry-run on the PC: RUNS_DIR is the instance's)
        cwd = REPO_ROOT
    py = REPO_ROOT / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    workers = args.workers or max(1, min(6, cpu_cores(env) // (2 * args.parallel)))

    jobs = []
    for spec in args.eval:
        src, out, crops = parse(spec, args.tag, runs_dir)
        ckpt = find_ckpt(src)
        if ckpt is None and not args.dry_run:
            sys.exit(f"ERROR: no checkpoint in {src / 'outputs'}")
        for n in crops:
            cmd = [py, "-m", "src.evaluate", ckpt or "<ckpt>", "--dataset-path", env["MUSDB_PROCESSED"], "-o", str(out / f"eval_crop{n}.json"),
                   "-w", str(workers), "--oracle-cache", env["ORACLE_CACHE_DIR"], "--crop-length", str(n)]
            jobs.append((out, n, cmd))

    def run(job):
        out, n, cmd = job
        target = out / f"eval_crop{n}.json"
        if target.exists() and not args.force:
            return f"{target.parent.name} crop {n}: already done, skipping"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"eval_crop{n}.log", "wb") as log:
            log.write(f"$ {shlex.join(cmd)}\n".encode()); log.flush()
            rc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            target.unlink(missing_ok=True)
        return f"{out.name} crop {n} : {'OK' if rc == 0 else f'FAILED (code {rc}), see {out / f'eval_crop{n}.log'}'}"

    print(f"{len(jobs)} evaluation(s), {workers} metric processes each, cwd {cwd}")
    if args.dry_run:
        for out, n, cmd in jobs:
            print(f"\n# {out.name} crop {n}\n  {shlex.join(cmd)}")
        return 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        results = list(pool.map(run, jobs))
    print("\n".join(results))
    return 1 if any("ECHEC" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
