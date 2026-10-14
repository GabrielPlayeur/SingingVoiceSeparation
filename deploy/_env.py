"""Lit deploy/env.sh (+ env.local.sh) : une seule source de vérité pour les scripts Python, sans dupliquer les valeurs."""
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
