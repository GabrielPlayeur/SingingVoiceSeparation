#!/usr/bin/env python3
"""Allège un checkpoint Lightning pour le transfert : garde les poids et les hyper-paramètres, retire l'état de l'optimiseur,
du scheduler et des callbacks (118 Mo -> ~39 Mo). Suffit pour `load_from_checkpoint`, l'évaluation et les analyses ; PAS pour
reprendre un entraînement. Usage : python deploy/slim_ckpt.py SRC.ckpt DST.ckpt
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
