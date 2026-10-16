"""Figures of the int8 experiment (experiments/08_int8_quantization.md) from the JSONs written by `python -m src.quantize`.

    python -m analysis.quantization --sweep outputs/quant/sweep_baseline.json --sensitivity outputs/quant/sensitivity_baseline.json \
        --qat outputs/quant/baseline.qat_p99_lr1e-5_cos.pt --qat outputs/quant/baseline.qat_p99_lr3e-5_cos.pt \
        --qat outputs/quant/baseline.qat_p99_lr1e-4_cos.pt --qat outputs/quant/baseline.qat_p99_lr3e-4_cos.pt \
        --qat outputs/quant/baseline.qat_minmax_lr1e-4_cos.pt --qat outputs/quant/baseline.qat_fp32_ctrl.pt

Figure labels are built from the stored configs (the setting names inside older JSONs are in French).
Outputs: figures/quant_ptq.png (PTQ ablations + per-quantizer sensitivity), figures/quant_qat.png (QAT curves).
All values are MEDIAN validation SI-SDRs (240 fixed crops, or 120 for the sensitivity): a fast measure to
compare settings; the test SDR (50 tracks) is measured with `python -m src.quantize evaluate`.
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from ._common import *

def lr_str(lr):
    return f"{lr:.0e}".replace("e-0", "e-")   # 1e-05 -> 1e-5


LR_COLORS = {1e-5: BLUE, 3e-5: ORANGE, 1e-4: AQUA, 3e-4: "#8a5fd1"}   # colour = learning rate (no palette cycle)


def sweep_label(cfg):
    """English label built from the config (the names in the JSON may be in French). With reduce_range, the activations are
    on 7 bits (0..127): written A7, as in experiments/08."""
    a = 7 if cfg["reduce_range"] else cfg["a_bits"]
    act = {"minmax": "min-max", "mse": "MSE", "percentile": f"p{cfg['percentile']:g}"}[cfg["act_range"]]
    if cfg["quantize_weights"] and not cfg["quantize_acts"]:
        detail = "per-tensor" if not cfg["w_per_channel"] else "ConvT per-channel too" if cfg["convt_per_channel"] else "per-channel"
        return f"weights only, W{cfg['w_bits']}, {detail}" + (", MSE range" if cfg["w_range"] == "mse" else "")
    head = f"activations only, A{a}" if not cfg["quantize_weights"] else f"W{cfg['w_bits']}A{a}"
    tail = ", weights MSE" if cfg["quantize_weights"] and cfg["w_range"] == "mse" else ""
    return f"{head}, {act}{tail}" + ("" if cfg["quantize_input"] else ", input in float")


def panel_sweep(ax, data):
    rows = [r for r in data["rows"] if r["config"]["quantize_weights"] or r["config"]["quantize_acts"]]   # without the fp32 control
    names = [sweep_label(r["config"]) for r in rows]
    vals = np.array([r["delta_median"] for r in rows])
    colors = [BLUE if not r["config"]["quantize_acts"] else ORANGE if not r["config"]["quantize_weights"] else AQUA for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.7, zorder=3)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Δ SI-SDR vs fp32 (dB)")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:+.2f}", (v, yi), textcoords="offset points", xytext=(-3 if v < 0 else 3, -3), ha="right" if v < 0 else "left", fontsize=7, color=INK_2)
    ax.set_title(f"PTQ ablations (fp32 {data['fp32']['median']:.2f} dB)")
    ax.set_xlim(min(vals.min() * 1.2, -0.2), 0.15)


def panel_sensitivity(ax, data):
    rows = data["rows"]
    labels = [f"{r['name']}" + ("" if r["kind"] == "activation" else " (weights)") for r in rows]
    vals = np.array([r["delta_median"] for r in rows])
    colors = [BLUE if r["kind"] == "activation" else ORANGE for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, vals, color=colors, zorder=3)
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_ylabel("Δ SI-SDR (dB)")
    ax.set_title("One quantizer at a time")
    ax.bar([0], [0], color=BLUE, label="activation"); ax.bar([0], [0], color=ORANGE, label="weights")
    ax.legend(loc="lower right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep")
    ap.add_argument("--sensitivity")
    ap.add_argument("--qat", action="append", default=[], help=".qat_*.pt file (repeatable)")
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
        fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
        ref = None
        ends = []
        for path in args.qat:
            d = torch.load(path, map_location="cpu", weights_only=False)
            h, cfg = d["history"], d["config"]
            ref = d["fp32_val"]["median"]
            control = not (cfg["quantize_weights"] or cfg["quantize_acts"])
            init = "min-max" if cfg["act_range"] == "minmax" else f"p{cfg['percentile']:g}"
            sched = "cosine" if d.get("cosine") else "constant"
            if control:
                label, color, ls, mk = f"fp32 control, LR {lr_str(d['lr'])}", INK_2, "--", "s"
            else:
                label, color, mk = f"QAT, {init}, LR {lr_str(d['lr'])}" + ("" if sched == "cosine" else ", constant"), \
                    LR_COLORS.get(d["lr"], INK), "o"
                ls = ":" if init == "min-max" else "-"
            xs, ys = [r["epoch"] for r in h], [r["val_median"] for r in h]
            ax.plot(xs, ys, marker=mk, markersize=5, color=color, linestyle=ls, linewidth=2, label=label)
            ends.append((ys[-1], xs[-1], label.split(",")[0] if control else f"{init}, {lr_str(d['lr'])}"))
        last = -np.inf   # direct labels at the end of the curves, spaced apart so they do not overlap
        for y, x, text in sorted(ends):
            last = max(y, last + 0.035)
            ax.text(x + 0.25, last, text, fontsize=7.5, color=INK_2, va="center")
        ax.axhline(ref, color=INK, linestyle="--", linewidth=1)
        ax.annotate(f"fp32 {ref:.2f} dB", (0, ref), textcoords="offset points", xytext=(2, 4), fontsize=8)
        ax.set_xlim(-0.3, max(r[1] for r in ends) + 2.2)
        ax.set_xlabel("QAT epoch"); ax.set_ylabel("validation SI-SDR (dB)")
        ax.set_title("Quantization-aware training, W8A7")
        ax.legend(fontsize=7.5, loc="lower right")
        save(fig, f"{args.out}_qat.png")


if __name__ == "__main__":
    main()
