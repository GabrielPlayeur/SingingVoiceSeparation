"""Figures de l'expérience int8 (experiments/08_int8_quantization.md) à partir des JSON écrits par `python -m src.quantize`.

    python -m analysis.quantization --sweep outputs/quant/sweep_baseline.json --sensitivity outputs/quant/sensitivity_baseline.json \
        --qat outputs/quant/baseline.qat_p99_lr1e-4.pt --qat outputs/quant/baseline.qat_p99_lr3e-5.pt

Sorties : figures/quant_ptq.png (ablations PTQ + sensibilité par quantificateur), figures/quant_qat.png (courbes de QAT).
Toutes les valeurs sont des SI-SDR MÉDIANS de validation (240 extraits fixes, ou 120 pour la sensibilité) : une mesure rapide pour
comparer des réglages ; le SDR de test (50 morceaux) se mesure avec `python -m src.quantize evaluate`.
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from ._common import *

PTQ_GROUPS = [("poids seuls", "poids seuls"), ("activations seules", "activations seules"), ("poids + activations", "W8A8")]


def panel_sweep(ax, data):
    rows = [r for r in data["rows"] if not r["name"].startswith("fp32")]
    names = [r["name"] for r in rows]
    vals = np.array([r["delta_median"] for r in rows])
    colors = [BLUE if n.startswith("poids seuls") else ORANGE if n.startswith("activations seules") else AQUA for n in names]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.7, zorder=3)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Δ SI-SDR médian de validation vs fp32 (dB)")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:+.2f}", (v, yi), textcoords="offset points", xytext=(-3 if v < 0 else 3, -3), ha="right" if v < 0 else "left", fontsize=7, color=INK_2)
    ax.set_title(f"Ablations PTQ (fp32 : {data['fp32']['median']:.2f} dB)")
    ax.set_xlim(min(vals.min() * 1.2, -0.2), 0.15)


def panel_sensitivity(ax, data):
    rows = data["rows"]
    labels = [f"{r['name']}" + ("" if r["kind"] == "activation" else " (poids)") for r in rows]
    vals = np.array([r["delta_median"] for r in rows])
    colors = [BLUE if r["kind"] == "activation" else ORANGE for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, vals, color=colors, zorder=3)
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_ylabel("Δ SI-SDR (dB)")
    ax.set_title("Un seul quantificateur à la fois (W8A8 minmax, les autres en flottant)")
    ax.bar([0], [0], color=BLUE, label="activation"); ax.bar([0], [0], color=ORANGE, label="poids")
    ax.legend(loc="lower right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep")
    ap.add_argument("--sensitivity")
    ap.add_argument("--qat", action="append", default=[], help="fichier .qat_*.pt (répétable)")
    ap.add_argument("--out", default="quant")
    args = ap.parse_args()
    setup_style()
    if args.sweep or args.sensitivity:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.15]})
        if args.sweep:
            panel_sweep(axes[0], json.load(open(args.sweep)))
        if args.sensitivity:
            panel_sensitivity(axes[1], json.load(open(args.sensitivity)))
        save(fig, f"{args.out}_ptq.png")
    if args.qat:
        import torch
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        colors = [BLUE, ORANGE, AQUA, INK_2, "#8a5fd1"]
        ref = ptq = None
        for i, path in enumerate(args.qat):
            d = torch.load(path, map_location="cpu", weights_only=False)
            h = d["history"]
            ref, ptq = d["fp32_val"]["median"], d["ptq_val"]["median"]
            label = path.split(".qat_")[-1].replace(".pt", "")
            ax.plot([r["epoch"] for r in h], [r["val_median"] for r in h], marker="o", color=colors[i % 5], linewidth=2, label=f"QAT {label}")
        ax.axhline(ref, color=INK, linestyle="--", linewidth=1); ax.annotate("fp32", (0, ref), textcoords="offset points", xytext=(2, 3), fontsize=8)
        ax.set_xlabel("époque de QAT (0 = PTQ de départ)"); ax.set_ylabel("SI-SDR médian de validation (dB)")
        ax.set_title("Quantization-aware training (W8A8)")
        ax.legend(fontsize=8)
        save(fig, f"{args.out}_qat.png")


if __name__ == "__main__":
    main()
