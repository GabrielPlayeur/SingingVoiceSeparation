#!/usr/bin/env python3
"""Slims a Lightning checkpoint for transfer: keeps the weights and the hyper-parameters, drops the optimizer, scheduler and
callbacks state (118 MB -> ~39 MB). Enough for `load_from_checkpoint`, the evaluation and the analyses; NOT to resume a
training. Usage: python deploy/slim_ckpt.py SRC DST.ckpt
SRC is a .ckpt file, or a run folder (RUNS_DIR/<tag>/<run>): the checkpoint chosen by its queue entry (best or last) is then used.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import find_ckpt  # noqa: E402

KEEP = ("state_dict", "hyper_parameters", "hparams_name", "datamodule_hyper_parameters", "datamodule_hparams_name",
        "pytorch-lightning_version", "epoch", "global_step")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = sys.argv[1]
    if Path(src).is_dir():
        src = find_ckpt(src) or sys.exit(f"no checkpoint in {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, sys.argv[2])
