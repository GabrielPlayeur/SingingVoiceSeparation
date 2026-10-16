"""Baseline vs oracle on the test set, from the per-track JSON written by `src.evaluate -o`.

Usage: python -m analysis.baseline_vs_oracle [outputs/oracle_baseline.json] [--tag baseline]
Outputs: figures/<tag>_vs_oracle.png, figures/<tag>_by_genre.png + text summary.
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", default="outputs/oracle_baseline.json")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    setup_style()

    with open(args.json) as f:
        res = json.load(f)
    meta = read_metadata()
    tracks = [r["track"] for r in res]
    sdr = np.array([r["sdr"] for r in res])
    ora = np.array([r["sdr_oracle"] for r in res])
    sis = np.array([r["si_sdri"] for r in res])
    sis_ora = np.array([r["si_sdri_oracle"] for r in res])
    gap = ora - sdr

    print(f"{len(res)} tracks")
    for name, v in (("SDR", sdr), ("SDR oracle", ora), ("SI-SDRi", sis), ("SI-SDRi oracle", sis_ora)):
        m, lo, hi = bootstrap_ci(v)
        print(f"{name:15s} median {m:6.2f} dB  95% CI [{lo:.2f}, {hi:.2f}]   mean {np.nanmean(v):.2f}")
    m, lo, hi = bootstrap_ci(gap)
    print(f"Oracle - model gap (SDR): median {m:.2f} dB 95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"Tracks where the model beats the oracle: {(gap < 0).sum()}/{len(res)}")
    rho, p = stats.spearmanr(sdr, ora)
    print(f"Model/oracle rank correlation (SDR): Spearman rho={rho:.2f} (p={p:.1e})")
    rho_g, p_g = stats.spearmanr(ora, gap)
    print(f"Oracle / gap correlation: Spearman rho={rho_g:.2f} (p={p_g:.1e})")
    print(f"Share of the oracle reached (median SDR / median oracle SDR): {np.nanmedian(sdr) / np.nanmedian(ora):.0%}")
    print(f"Paired Wilcoxon test model vs oracle: p={stats.wilcoxon(sdr, ora, nan_policy='omit').pvalue:.1e}")
    order = np.argsort(-gap)
    print("Largest gaps:", [(tracks[i], round(gap[i], 1)) for i in order[:5]])
    print("Smallest gaps:", [(tracks[i], round(gap[i], 1)) for i in order[-3:]])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    # (1) model vs oracle scatter
    ax = axes[0]
    ax.scatter(ora, sdr, color=BLUE, s=34, edgecolor="white", linewidth=0.8, zorder=3)
    lim = [min(np.nanmin(ora), np.nanmin(sdr)) - 1, max(np.nanmax(ora), np.nanmax(sdr)) + 1]
    ax.plot(lim, lim, color=INK_2, linestyle="--", linewidth=1)
    ax.text(lim[1] - 0.3, lim[1] - 1.2, "model = oracle", ha="right", color=INK_2, fontsize=9)
    for i in order[:3]:
        ax.annotate(artist(tracks[i]) + " – " + tracks[i].split(" - ")[1][:14], (ora[i], sdr[i]),
                    xytext=(6, -4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SDR oracle (dB)"); ax.set_ylabel("model SDR (dB)")
    ax.set_title(f"Per track (Spearman ρ = {rho:.2f})")

    # (2) distributions of the 4 metrics
    ax = axes[1]
    groups = [("SDR\nmodel", sdr, BLUE), ("SDR\noracle", ora, ORANGE), ("SI-SDRi\nmodel", sis, BLUE), ("SI-SDRi\noracle", sis_ora, ORANGE)]
    pos = [0, 1, 2.6, 3.6]
    for x, (lab, v, c) in zip(pos, groups):
        v = v[~np.isnan(v)]
        bp = ax.boxplot(v, positions=[x], widths=0.6, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=2), whiskerprops=dict(color=INK_2),
                        capprops=dict(color=INK_2), flierprops=dict(markersize=3, markeredgecolor=INK_2))
        bp["boxes"][0].set(facecolor=c, alpha=0.55, edgecolor=c)
        m, lo, hi = bootstrap_ci(v)
        ax.annotate(f"{m:.1f}", (x + 0.38, m), fontsize=9, color=INK, va="center")
    ax.set_xticks(pos, [g[0] for g in groups])
    ax.set_ylabel("dB")
    ax.set_title("Distribution over the tracks (median annotated)")

    # (3) per-track gap, sorted
    ax = axes[2]
    sorted_gap = gap[order[::-1]]
    ax.bar(np.arange(len(res)), sorted_gap, color=np.where(sorted_gap >= 0, BLUE, ORANGE), width=0.8)
    ax.axhline(np.median(gap), color=INK, linewidth=1.5)
    ax.text(0, np.median(gap) + 0.25, f"median {np.median(gap):.1f} dB", ha="left", color=INK_2)
    ax.set_xlabel("tracks, sorted"); ax.set_ylabel("oracle SDR − model SDR (dB)")
    ax.set_title("Room for improvement per track")
    fig.suptitle(f"{args.tag} vs oracle (ideal mask bounded to [0, 1], mixture phase)", fontweight="bold")
    save(fig, f"{args.tag}_vs_oracle.png")

    # (4) per genre
    genres = np.array([meta[t]["genre"] for t in tracks])
    uniq = [g for g in sorted(set(genres), key=lambda g: -np.nanmedian(sdr[genres == g])) if (genres == g).sum() >= 2]
    print("\nGenre (n ≥ 2)        n   SDR   oracle  gap")
    for g in uniq:
        k = genres == g
        print(f"{g:18s} {k.sum():3d}  {np.nanmedian(sdr[k]):5.2f}  {np.nanmedian(ora[k]):5.2f}  {np.nanmedian(gap[k]):5.2f}")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(uniq))
    for yi, g in zip(y, uniq):
        k = genres == g
        ax.plot([np.nanmedian(sdr[k]), np.nanmedian(ora[k])], [yi, yi], color=GRID, linewidth=4, zorder=1, solid_capstyle="round")
        ax.scatter(sdr[k], np.full(k.sum(), yi), color=BLUE, s=14, alpha=0.35, zorder=2)
        ax.scatter(np.nanmedian(sdr[k]), yi, color=BLUE, s=70, edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter(np.nanmedian(ora[k]), yi, color=ORANGE, s=70, edgecolor="white", linewidth=1.2, zorder=4)
    ax.set_yticks(y, [f"{g} (n={(genres == g).sum()})" for g in uniq]); ax.invert_yaxis()
    ax.set_xlabel("SDR (dB): median per genre, points = tracks of the model")
    ax.scatter([], [], color=BLUE, s=50, label="model"); ax.scatter([], [], color=ORANGE, s=50, label="oracle")
    ax.legend(loc="lower right")
    ax.set_title("SDR per genre")
    save(fig, f"{args.tag}_by_genre.png")


if __name__ == "__main__":
    main()
