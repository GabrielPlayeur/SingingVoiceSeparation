#!/usr/bin/env python3
"""Évalue des checkpoints déjà entraînés à d'AUTRES longueurs de fenêtre d'inférence (expérience « longueur de contexte »,
experiments/07_context_length.md). Rien n'est réentraîné : `src.evaluate --crop-length N` sur le meilleur checkpoint du run.

    python deploy/eval_crops.py --tag ctx-v1 --eval ctx3_s42=130560 --eval ctx12_s42=523776 \
        --eval loss-v1/baseline_s42=130560,392704,523776 [--parallel 2] [--dry-run]

- `--eval RUN=N[,N...]` : RUN est un dossier de RUNS_DIR/<tag>/ ; `AUTRE_TAG/RUN` désigne un run d'un autre tag (ex. la baseline de
  loss-v1) : ses résultats vont dans RUNS_DIR/<tag>/infer_<AUTRE_TAG>_<RUN>/ pour partir avec le rapatriement de <tag>.
- Sortie : eval_crop<N>.json (+ .log) à côté de eval.json, même format. Déjà présent : saute (--force pour refaire).
- N est en échantillons et doit donner un nombre de trames STFT multiple de 2**depth : 130560 (2,96 s), 261632 (5,93 s),
  392704 (8,90 s), 523776 (11,88 s).
- Le code exécuté est le checkout figé de <tag> (RUNS_DIR/<tag>/_code, créé par run_queue.py) : il doit contenir `--crop-length`.
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
from _env import REPO_ROOT, cpu_cores, load_env


def find_ckpt(rundir):
    found = glob.glob(str(rundir / "outputs" / "VoiceSeparation" / "*" / "checkpoints" / "*.ckpt"))
    return max(found, key=os.path.getmtime) if found else None


def parse(spec, tag, runs_dir):
    run, _, crops = spec.partition("=")
    if not crops:
        sys.exit(f"ERREUR: --eval {spec!r} : attendu RUN=N[,N...]")
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
    ap.add_argument("--parallel", type=int, default=1, help="évaluations simultanées (défaut 1)")
    ap.add_argument("--workers", type=int, default=None, help="processus de métriques par évaluation (défaut : cœurs / (2 x parallel), 6 au plus)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(env["RUNS_DIR"])
    code = runs_dir / args.tag / "_code"
    try:
        cwd = code if code.exists() else REPO_ROOT
    except OSError:   # dossier inaccessible (ex. --dry-run sur le PC : RUNS_DIR est celui de l'instance)
        cwd = REPO_ROOT
    py = REPO_ROOT / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    workers = args.workers or max(1, min(6, cpu_cores(env) // (2 * args.parallel)))

    jobs = []
    for spec in args.eval:
        src, out, crops = parse(spec, args.tag, runs_dir)
        ckpt = find_ckpt(src)
        if ckpt is None and not args.dry_run:
            sys.exit(f"ERREUR: aucun checkpoint dans {src / 'outputs'}")
        for n in crops:
            cmd = [py, "-m", "src.evaluate", ckpt or "<ckpt>", "--dataset-path", env["MUSDB_PROCESSED"], "-o", str(out / f"eval_crop{n}.json"),
                   "-w", str(workers), "--oracle-cache", env["ORACLE_CACHE_DIR"], "--crop-length", str(n)]
            jobs.append((out, n, cmd))

    def run(job):
        out, n, cmd = job
        target = out / f"eval_crop{n}.json"
        if target.exists() and not args.force:
            return f"{target.parent.name} crop {n} : déjà fait, saute"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"eval_crop{n}.log", "wb") as log:
            log.write(f"$ {shlex.join(cmd)}\n".encode()); log.flush()
            rc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            target.unlink(missing_ok=True)
        return f"{out.name} crop {n} : {'OK' if rc == 0 else f'ECHEC (code {rc}), voir {out / f'eval_crop{n}.log'}'}"

    print(f"{len(jobs)} évaluation(s), {workers} processus de métriques chacune, cwd {cwd}")
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
