"""Longueur de contexte : SDR de test en fonction de la longueur des extraits (entraînement) ou des fenêtres (inférence).

Trois séries, qui séparent deux effets (experiments/07_context_length.md) :
  --train  SECONDES=eval.json : modèle ENTRAÎNÉ sur des extraits de SECONDES, évalué avec les fenêtres de la baseline (5,93 s).
                                Mesure l'effet de l'entraînement ; l'inférence est la même pour tous. Série principale.
  --own    SECONDES=eval.json : même modèle, évalué avec des fenêtres de SECONDES (= sa longueur d'entraînement).
                                Entraînement + inférence longs ensemble, comme dans HT Demucs (Rouard et al. 2023, § 5.1).
  --infer  SECONDES=eval.json : la checkpoint de la BASELINE (entraînée sur 5,93 s), évaluée avec des fenêtres de SECONDES.
                                Mesure l'effet de l'inférence seule, sans réentraîner.
  --control SECONDES=eval.json : un point de contrôle de la série --train (ex. batch 4 : même audio par pas que la baseline).
La référence des tests appariés est le point `--ref` (5,93 s par défaut) de la série --train : la baseline.

Usage :
    python -m analysis.context_length \
        --train 2.96=outputs/server/ctx-v1/ctx3_s42/eval.json --train 5.93=outputs/server/loss-v1/baseline_s42/eval.json \
        --train 8.90=outputs/server/ctx-v1/ctx9_s42/eval.json --train 11.88=outputs/server/ctx-v1/ctx12_s42/eval.json \
        --control 11.88=outputs/server/ctx-v1/ctx12_b4_s42/eval.json \
        --own 2.96=outputs/server/ctx-v1/ctx3_s42/eval_crop130560.json --own 11.88=outputs/server/ctx-v1/ctx12_s42/eval_crop523776.json \
        --infer 2.96=outputs/server/ctx-v1/infer_loss-v1_baseline_s42/eval_crop130560.json --infer 11.88=...eval_crop523776.json
Sorties : figures/context_length.png, figures/context_length.json
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *

SHIFT = {"train": 1.0, "own": 1.03, "infer": 0.97, "control": 1.06}   # décalage horizontal pour que les barres d'erreur ne se superposent pas
RF_SECONDS = 4.42   # champ récepteur total mesuré du U-Net (381 trames STFT de 512 échantillons à 44,1 kHz), experiments/07_context_length.md


def load(path):
    with open(path) as f:
        return {r["track"]: r for r in json.load(f)}


def pairs(specs):
    out = []
    for s in specs:
        sec, _, path = s.partition("=")
        out.append((float(sec), path))
    return sorted(out)


def row(sec, path, common, ref):
    rows = load(path)
    tr = sorted(common)
    r = {"seconds": sec, "path": path}
    for key in ("sdr", "si_sdri"):
        m, lo, hi = bootstrap_ci([rows[t][key] for t in tr])
        r[key] = {"median": m, "lo": lo, "hi": hi}
    a = np.array([ref[t]["sdr"] for t in tr]); b = np.array([rows[t]["sdr"] for t in tr])
    d = b - a
    m, lo, hi = bootstrap_ci(d)
    r["vs_ref"] = {"median_diff": m, "lo": lo, "hi": hi, "wins": int((d > 0).sum()),
                   "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0}
    return r


def main():
    ap = argparse.ArgumentParser()
    for name in ("train", "own", "infer", "control"):
        ap.add_argument(f"--{name}", action="append", default=[], metavar="SECONDES=eval.json")
    ap.add_argument("--ref", type=float, default=5.93, help="point de la série --train qui sert de baseline (défaut 5.93)")
    ap.add_argument("--out", default="context_length")
    args = ap.parse_args()
    if len(args.train) < 2:
        ap.error("au moins deux --train SECONDES=eval.json, dont la baseline")
    setup_style()

    series = {n: pairs(getattr(args, n)) for n in ("train", "own", "infer", "control")}
    ref_path = dict(series["train"]).get(args.ref)
    if ref_path is None:
        ap.error(f"pas de --train {args.ref}=... (la baseline sert de référence)")
    ref = load(ref_path)
    common = set(ref)
    for s in series.values():
        for _, p in s:
            common &= set(load(p))
    res = {n: [row(sec, p, common, ref) for sec, p in s] for n, s in series.items()}

    print(f"{len(common)} morceaux de test communs ; référence : baseline entraînée et évaluée sur {args.ref} s")
    labels = {"train": "entraîné sur L, évalué sur 5,93 s", "own": "entraîné sur L, évalué sur L",
              "infer": "baseline évaluée sur L (sans réentraîner)", "control": "contrôle (même audio par pas)"}
    for n, rows in res.items():
        if not rows:
            continue
        print(f"\n[{n}] {labels[n]}")
        print(f"  {'L (s)':>7s}  {'SDR médian [IC95 %]':24s}{'SI-SDRi':>9s}   vs baseline (SDR apparié)")
        for r in rows:
            v = r["vs_ref"]; s = r["sdr"]
            print(f"  {r['seconds']:7.2f}  {s['median']:6.2f} [{s['lo']:.2f}, {s['hi']:.2f}]        {r['si_sdri']['median']:6.2f}   "
                  f"{v['median_diff']:+.2f} dB [{v['lo']:+.2f}, {v['hi']:+.2f}], meilleur sur {v['wins']}/{len(common)}, p={v['wilcoxon_p']:.2g}")
    tr = res["train"]
    if len(tr) >= 3:
        x = np.log2([r["seconds"] for r in tr]); y = [r["sdr"]["median"] for r in tr]
        slope = float(np.polyfit(x, y, 1)[0])
        print(f"\nPente de la série --train : {slope:+.2f} dB par doublement de la longueur d'extrait")
    else:
        slope = None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    style = {"train": (BLUE, "o", "-"), "own": (ORANGE, "s", "--"), "infer": (AQUA, "^", ":"), "control": (INK, "D", "none")}
    for ax, key, title in ((axes[0], "sdr", "SDR médian (museval)"), (axes[1], "si_sdri", "SI-SDRi médian")):
        for n, rows in res.items():
            if not rows:
                continue
            c, mk, ls = style[n]
            xs = [r["seconds"] * SHIFT[n] for r in rows]; ys = [r[key]["median"] for r in rows]
            ax.errorbar(xs, ys, yerr=[[r[key]["median"] - r[key]["lo"] for r in rows], [r[key]["hi"] - r[key]["median"] for r in rows]],
                        color=c, marker=mk, linestyle=ls, linewidth=2, markersize=7, capsize=3, label=labels[n], zorder=3)
        ax.axvline(RF_SECONDS, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate(f"champ récepteur {RF_SECONDS} s", (RF_SECONDS, ax.get_ylim()[0]), textcoords="offset points", xytext=(4, 6), fontsize=8, color=INK_2, va="bottom")
        ax.set_xscale("log", base=2)
        ticks = sorted({r["seconds"] for rows in res.values() for r in rows})
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlabel("longueur L (s, échelle log)"); ax.set_ylabel("dB"); ax.set_title(title)
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Longueur de contexte sur {len(common)} morceaux de test (barres : IC95 % bootstrap de la médiane, une graine)", fontweight="bold")
    save(fig, f"{args.out}.png")
    with open(f"{FIG_DIR}/{args.out}.json", "w") as f:
        json.dump({"tracks": len(common), "ref_seconds": args.ref, "slope_db_per_doubling": slope, "receptive_field_s": RF_SECONDS, **res}, f, indent=2)
    print(f"-> figures/{args.out}.json")


if __name__ == "__main__":
    main()
