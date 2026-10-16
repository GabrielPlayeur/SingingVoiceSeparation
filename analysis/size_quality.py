"""Size / quality curve: test SDR as a function of the number of parameters, when only the U-Net width changes.

Hypothesis tested (README, section 5): a bigger model separates better, at a fixed training budget (150 epochs). The width
is `model.base_channels` (depth 6 and data unchanged); the number of parameters is recounted here by instantiating
`src.models.unet.UNet` (not read from a checkpoint: the server's eval.json files do not contain it).

Usage:
    python -m analysis.size_quality \
        --run 4=outputs/server/size-v1/w4_s42/eval.json  --run 8=outputs/server/size-v1/w8_s42/eval.json \
        --run 16=outputs/server/size-v1/w16_s42/eval.json --run 32=outputs/server/size-v1/w32_s42/eval.json \
        --run 64=outputs/server/size-v1/w64_s42/eval.json \
        --variant power_loss=outputs/server/power-v1/power_loss_s42/eval.json
        --rerun local=outputs/eval_stereo.json --rerun loss-v1=outputs/server/loss-v1/baseline_s42/eval.json
`--run WIDTH=eval.json` (base_channels) defines the curve. `--rerun NAME=eval.json` adds other trainings of the reference
width (same config, same seed): their spread shows the noise of a retraining at this width. `--variant NAME=eval.json[:WIDTH]`
places another model on the same figure at the size of its width (16 by default: same parameters as the baseline, so it is read
vertically above the 16 point).
The eval.json files come from `src.evaluate -o` (50 test tracks, fields `sdr`, `si_sdri`, `sdr_oracle`).
Outputs: figures/size_quality.png, figures/size_quality.json
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *


def load(path):
    with open(path) as f:
        return {r["track"]: r for r in json.load(f)}


def n_params(width, depth, in_channels):
    from src.models.unet import UNet
    return sum(p.numel() for p in UNet(width, depth, in_channels).parameters())


def parse(spec, with_width):
    name, _, rest = spec.partition("=")
    path, _, width = rest.partition(":") if with_width else (rest, "", "")
    return name, path, int(width) if width else None


def stat_row(label, width, path, tracks_ref, ref, depth, in_channels, metric_keys):
    rows = load(path)
    tracks = sorted(set(rows) & tracks_ref)
    p = n_params(width, depth, in_channels)
    row = {"label": label, "width": width, "params": p, "fp32_mib": p * 4 / 2**20, "n_tracks": len(tracks), "path": path}
    for key in metric_keys:
        v = np.array([rows[t][key] for t in tracks])
        m, lo, hi = bootstrap_ci(v)
        row[key] = {"median": m, "lo": lo, "hi": hi}
    if ref is not None:
        a = np.array([ref[t]["sdr"] for t in tracks]); b = np.array([rows[t]["sdr"] for t in tracks])
        d = b - a
        m, lo, hi = bootstrap_ci(d)
        row["vs_ref"] = {"median_diff": m, "lo": lo, "hi": hi, "wins": int((d > 0).sum()),
                         "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0}
    return row, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="WIDTH=eval.json", help="one point of the curve (base_channels)")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=eval.json[:WIDTH]", help="another model placed on the curve")
    ap.add_argument("--rerun", action="append", default=[], metavar="NAME=eval.json", help="another training of the --ref width")
    ap.add_argument("--ref", type=int, default=16, help="width of the baseline, reference of the paired tests (default 16)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--in-channels", type=int, default=2, help="2 = stereo")
    ap.add_argument("--out", default="size_quality")
    args = ap.parse_args()
    if len(args.run) < 2:
        ap.error("at least two --run WIDTH=eval.json")
    setup_style()

    runs = sorted((int(w), p) for w, p, _ in (parse(s, False) for s in args.run))
    widths = [w for w, _ in runs]
    if len(set(widths)) != len(widths):
        ap.error("duplicate widths in --run")
    common = set.intersection(*(set(load(p)) for _, p in runs))
    ref = load(dict(runs)[args.ref]) if args.ref in widths else None
    if ref is None:
        print(f"WARNING: no --run {args.ref}=...: no paired test against the baseline")
    keys = ("sdr", "si_sdri")

    curve = [stat_row(f"w{w}", w, p, common, ref, args.depth, args.in_channels, keys)[0] for w, p in runs]
    variants = []
    for s in args.variant:
        name, path, width = parse(s, True)
        variants.append(stat_row(name, width or args.ref, path, common, ref, args.depth, args.in_channels, keys)[0])
    reruns = []
    for s in args.rerun:
        name, path, _ = parse(s, False)
        reruns.append(stat_row(name, args.ref, path, common, ref, args.depth, args.in_channels, keys)[0])
    oracle_rows = load(runs[0][1])
    oracle = bootstrap_ci([oracle_rows[t]["sdr_oracle"] for t in sorted(common)])[0]

    print(f"{len(common)} common test tracks; oracle (ceiling of the real mask with the mixture phase): {oracle:.2f} dB")
    print(f"{'model':12s}{'width':>8s}{'params':>10s}{'fp32 MiB':>10s}   {'median SDR [95% CI]':24s}{'SI-SDRi':>9s}   vs w{args.ref} (SDR)")
    for r in curve + reruns + variants:
        s = r["sdr"]
        vs = r.get("vs_ref")
        tail = f"{vs['median_diff']:+.2f} dB [{vs['lo']:+.2f}, {vs['hi']:+.2f}], better on {vs['wins']}/{r['n_tracks']}, p={vs['wilcoxon_p']:.2g}" if vs else ""
        print(f"{r['label']:12s}{r['width']:8d}{r['params'] / 1e6:9.2f}M{r['fp32_mib']:10.1f}   "
              f"{s['median']:6.2f} [{s['lo']:.2f}, {s['hi']:.2f}]        {r['si_sdri']['median']:6.2f}   {tail}")

    # slope: dB gained at each doubling of the width (x4 in parameters), linear fit on log2(width)
    x = np.log2([r["width"] for r in curve]); y = np.array([r["sdr"]["median"] for r in curve])
    slope, intercept = np.polyfit(x, y, 1)
    steps = np.diff(y)
    print(f"Overall slope: {slope:+.2f} dB per width doubling (x4 in parameters); gain between consecutive points: "
          + ", ".join(f"{curve[i]['label']}→{curve[i + 1]['label']} {d:+.2f}" for i, d in enumerate(steps)))
    print("Courbe monotone" if np.all(steps > 0) else "WARNING: the curve is not monotonic (a bigger model does worse: under-training?)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for ax, key, title in ((axes[0], "sdr", "Median test SDR (museval)"), (axes[2], "si_sdri", "Median test SI-SDRi")):
        px = [r["params"] for r in curve]
        ax.errorbar(px, [r[key]["median"] for r in curve],
                    yerr=[[r[key]["median"] - r[key]["lo"] for r in curve], [r[key]["hi"] - r[key]["median"] for r in curve]],
                    color=BLUE, marker="o", markersize=7, linewidth=2, capsize=3, label="U-Net width (base_channels)", zorder=3)
        if reruns:
            ax.scatter([r["params"] for r in reruns], [r[key]["median"] for r in reruns], s=60, facecolor="white", edgecolor=BLUE,
                       linewidth=1.5, zorder=4, label=f"other trainings of w{args.ref} (same config and seed)")
        for r in curve:
            ax.annotate(r["label"], (r["params"], r[key]["median"]), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color=INK_2)
        for i, r in enumerate(variants):
            ax.errorbar([r["params"]], [r[key]["median"]], yerr=[[r[key]["median"] - r[key]["lo"]], [r[key]["hi"] - r[key]["median"]]],
                        color=[ORANGE, AQUA, INK_2][i % 3], marker="D", markersize=7, linestyle="none", capsize=3, label=r["label"], zorder=4)
        if key == "sdr":
            ax.axhline(oracle, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"oracle {oracle:.2f} dB", (px[0], oracle), textcoords="offset points", xytext=(0, 4), fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (log scale); fp32 weights: " + ", ".join(f"{r['label']} {r['fp32_mib']:.0f} MiB" for r in curve), fontsize=8)
        ax.set_ylabel("dB"); ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
    if ref is not None:  # central panel: paired difference per track against the reference width (the measure that settles it)
        ax = axes[1]
        def paired_err(rows, x, **kw):
            m = [r["vs_ref"]["median_diff"] for r in rows]
            ax.errorbar(x, m, yerr=[[a - r["vs_ref"]["lo"] for a, r in zip(m, rows)], [r["vs_ref"]["hi"] - a for a, r in zip(m, rows)]],
                        capsize=3, **kw)
        paired_err(curve, [r["params"] for r in curve], color=BLUE, marker="o", markersize=7, linewidth=2, zorder=3,
                   label=f"U-Net width vs w{args.ref}")
        if reruns:
            paired_err(reruns, [r["params"] * (1.12 + 0.12 * i) for i, r in enumerate(reruns)], color=BLUE, marker="o", markersize=7,
                       markerfacecolor="white", linestyle="none", zorder=4, label=f"other trainings of w{args.ref} (rerun noise)")
        for i, r in enumerate(variants):  # variants (loss, compression) to the left of the reference point
            paired_err([r], [r["params"] / (1.12 + 0.12 * i)], color=[ORANGE, AQUA, INK_2][i % 3], marker="D", markersize=6,
                       linestyle="none", zorder=4, label=f"{r['label']} (w{r['width']})")
        for r in curve:
            ax.annotate(f"{r['label']}  {r['vs_ref']['median_diff']:+.2f}", (r["params"], r["vs_ref"]["median_diff"]),
                        textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color=INK_2)
        ax.axhline(0, color=INK_2, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.margins(x=0.12)
        ax.set_xlabel("parameters (log scale)")
        ax.set_ylabel(f"SDR difference vs w{args.ref}, paired per track (dB)")
        ax.set_title(f"Paired SDR difference vs w{args.ref} (median, 95% CI)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Size / quality on {len(common)} test tracks (bars: 95% bootstrap CI of the median; one seed per width)",
                 fontweight="bold")
    save(fig, f"{args.out}.png")
    with open(f"{FIG_DIR}/{args.out}.json", "w") as f:
        json.dump({"tracks": len(common), "oracle_sdr": oracle, "slope_db_per_width_doubling": slope, "curve": curve, "reruns": reruns,
                   "variants": variants},
                  f, indent=2)
    print(f"-> figures/{args.out}.json")


if __name__ == "__main__":
    main()
