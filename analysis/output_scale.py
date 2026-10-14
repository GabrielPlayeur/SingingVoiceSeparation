"""Niveau de la voix estimée : le modèle sort-il la bonne échelle ? (ROADMAP 2b, Le Roux et al. 2019 § 2.3)

Le SI-SDR ne pénalise pas une erreur de gain (il re-met la cible à l'échelle optimale, voir `scale_error`),
alors que le SDR de l'évaluation (museval) et le SNR en dépendent. Une loss -SI-SDR ne donne donc aucun signal sur le niveau de
sortie. Comme le masque est borné dans [0, 1], la dérive possible est une voix trop faible (α > 1).

Ce script mesure, par extrait de validation et pour chaque checkpoint :
  - SI-SDR et SNR (dB) de la voix estimée ;
  - le gain optimal α (en dB ; > 0 : voix estimée trop faible, < 0 : trop forte) ;
  - `coût de l'échelle` = SNR(α·ŝ) - SNR(ŝ) >= 0 (dB) : ce que gagnerait le SNR si la voix estimée avait le bon niveau ;
    0 = niveau parfait. (Attention : SI-SDR - SNR n'est PAS ce coût ; SI-SDR n'est pas borné par le SNR, voir Le Roux § 2.2.)

Usage : python -m analysis.output_scale [--ckpt NOM=CHEMIN ...] [--crops-per-track 5]
Sans --ckpt : la baseline stéréo (run fgvzjxua). Sorties : figures/output_scale.png, figures/output_scale.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from ._common import *
from .input_amplitude import DEFAULT_CKPTS, load_val_crops
from src.module import VoiceSeparationModule
from src.utils.utils import si_sdr


def scale_error(est, target):
    """est, target (B, C, T) -> SI-SDR, SNR, gain optimal α (dB) et coût de l'échelle (dB), moyennés sur les canaux.

    SNR = 10 log10(‖s‖² / ‖s - ŝ‖²) est l'échelle prise « telle quelle » (Le Roux et al., eq. 1) ;
    α = <ŝ, s> / <ŝ, ŝ> : facteur par lequel il faut multiplier ŝ pour qu'il s'aligne sur s."""
    snr = lambda e: 10 * torch.log10(target.pow(2).sum(-1) / (target - e).pow(2).sum(-1).clamp_min(1e-12))
    alpha = ((est * target).sum(-1) / est.pow(2).sum(-1).clamp_min(1e-12)).clamp_min(1e-6)  # (B, C)
    cost = snr(alpha[..., None] * est) - snr(est)
    return si_sdr(est, target), snr(est).mean(-1), (20 * torch.log10(alpha)).mean(-1), cost.mean(-1)


@torch.no_grad()
def measure(module, mixture, vocals, device, batch_size=8):
    module = module.to(device).eval()
    out = {"si_sdr": [], "snr": [], "alpha_db": [], "cost": []}
    for i in range(0, len(mixture), batch_size):
        mix, voc = mixture[i:i + batch_size].to(device), vocals[i:i + batch_size].to(device)
        if module.hparams.in_channels == 1:  # ancien modèle mono : appliqué à L puis R
            est = torch.cat([module.predict_step(module.on_after_batch_transfer({"mixture": mix[:, c:c + 1]}, 0), 0) for c in range(2)], 1)
        else:
            est = module.predict_step(module.on_after_batch_transfer({"mixture": mix}, 0), 0)
        for k, v in zip(out, scale_error(est, voc)):
            out[k].append(v.cpu())
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], metavar="NOM=CHEMIN")
    ap.add_argument("--crops-per-track", type=int, default=5)
    args = ap.parse_args()
    setup_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    if not ckpts:
        for name, run in DEFAULT_CKPTS.items():
            found = glob.glob(os.path.join(ROOT, f"outputs/VoiceSeparation/{run}/checkpoints/*.ckpt"))
            if found:
                ckpts[name] = found[0]

    mixture, vocals = load_val_crops(args.crops_per_track)
    print(f"{len(mixture)} extraits de validation")
    results = {}
    for name, path in ckpts.items():
        r = measure(VoiceSeparationModule.load_from_checkpoint(path, map_location="cpu"), mixture, vocals, device)
        results[name] = r
        print(f"{name:14s} SI-SDR {np.median(r['si_sdr']):5.2f} dB | SNR {np.median(r['snr']):5.2f} dB | coût de l'échelle médian "
              f"{np.median(r['cost']):.2f} dB (p90 {np.percentile(r['cost'], 90):.2f}) | gain optimal médian {np.median(r['alpha_db']):+.2f} dB "
              f"(p10 {np.percentile(r['alpha_db'], 10):+.2f}, p90 {np.percentile(r['alpha_db'], 90):+.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    colors = [BLUE, ORANGE, AQUA]
    for i, (name, r) in enumerate(results.items()):
        c = colors[i % len(colors)]
        axes[0].hist(r["cost"], bins=np.linspace(0, max(3, np.percentile(r["cost"], 99)), 40), alpha=0.55, color=c, label=name)
        axes[1].hist(r["alpha_db"], bins=np.linspace(-10, 10, 41), alpha=0.55, color=c, label=name)
    axes[0].set_xlabel("coût de l'échelle : SNR(α·ŝ) − SNR(ŝ)  (dB)")
    axes[0].set_ylabel("extraits de validation")
    axes[0].set_title("Ce que coûte le niveau de sortie")
    axes[1].axvline(0, color=INK_2, linewidth=1, linestyle="--")
    axes[1].set_xlabel("gain optimal α de la voix estimée (dB)   [> 0 : estimée trop faible]")
    axes[1].set_title("Niveau de la voix estimée")
    axes[0].legend()
    save(fig, "output_scale.png")
    with open(os.path.join(FIG_DIR, "output_scale.json"), "w") as f:
        json.dump({n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in results.items()}, f)


if __name__ == "__main__":
    main()
