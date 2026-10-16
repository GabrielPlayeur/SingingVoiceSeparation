"""Reads deploy/env.sh (+ env.local.sh): a single source of truth for the Python scripts, without duplicating the values."""
import glob
import json
import os
import subprocess
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent


def load_env():
    out = subprocess.run(["bash", "-c", f'set -a; source "{DEPLOY_DIR / "env.sh"}"; env -0'],
                         check=True, capture_output=True, text=True).stdout
    env = dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
    return env


def cpu_cores(env):
    if env.get("CPU_CORES"):
        return int(env["CPU_CORES"])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def mem_total_gb():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("MemTotal")) / 1e6


def find_ckpt(rundir, kind=None):
    """Checkpoint of a run to evaluate. `kind="best"`: the best-on-validation .ckpt (the most recent one that is neither last*.ckpt
    nor periodic-*); `kind="last"`: last.ckpt (final weights, for fixed-budget runs). `kind=None`: the `ckpt` recorded by
    run_queue.py in the run's metadata.json, "best" if absent."""
    rundir = Path(rundir)
    if kind is None:
        try:
            kind = json.loads((rundir / "metadata.json").read_text()).get("ckpt", "best")
        except (OSError, ValueError):
            kind = "best"
    found = glob.glob(str(rundir / "outputs" / "VoiceSeparation" / "*" / "checkpoints" / "*.ckpt"))
    if kind == "last":
        found = [c for c in found if os.path.basename(c).startswith("last")]
    elif kind == "best":
        found = [c for c in found if not os.path.basename(c).startswith(("last", "periodic"))]
    else:
        raise ValueError(f"ckpt kind {kind!r}: expected 'best' or 'last'")
    return max(found, key=os.path.getmtime) if found else None
