#!/usr/bin/env python3
"""Slims a Lightning checkpoint for transfer: keeps the weights and the hyper-parameters, drops the optimizer, scheduler and
callbacks state (118 MB -> ~39 MB). Enough for `load_from_checkpoint`, the evaluation and the analyses; NOT to resume a
training. Usage: python deploy/slim_ckpt.py SRC.ckpt DST.ckpt
"""
import sys

import torch

KEEP = ("state_dict", "hyper_parameters", "hparams_name", "datamodule_hyper_parameters", "datamodule_hparams_name",
        "pytorch-lightning_version", "epoch", "global_step")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, sys.argv[2])
