"""Context length: test SDR as a function of the crop length (training) or of the window length (inference).

Three series, which separate two effects (experiments/07_context_length.md):
  --train  SECONDS=eval.json : model TRAINED on crops of SECONDS, evaluated with the baseline's windows (5.93 s).
                               Measures the effect of training; inference is the same for all. Main series.
  --own    SECONDS=eval.json : same model, evaluated with windows of SECONDS (= its training length).
                               Long training + long inference together, as in HT Demucs (Rouard et al. 2023, § 5.1).
  --infer  SECONDS=eval.json : the BASELINE checkpoint (trained on 5.93 s), evaluated with windows of SECONDS.
                               Measures the effect of inference alone, without retraining.
  --control SECONDS=eval.json : a control point of the --train series (e.g. batch 4: same audio per step as the baseline).
  --rerun  NAME=eval.json     : another training of the baseline (same config, same seed): noise of a retraining.
The reference of the paired tests is the `--ref` point (5.93 s by default) of the --train series: the baseline.
Comparisons are read on the PAIRED differences per track (panels a and b): the median of a single run moves by
~0.3 dB from one retraining to the next (experiments/06), the paired difference by ±0.15 dB.

Usage:
    python -m analysis.context_length \
        --train 2.96=outputs/server/ctx-v1/ctx3_s42/eval.json --train 5.93=outputs/server/loss-v1/baseline_s42/eval.json \
        --train 8.90=outputs/server/ctx-v1/ctx9_s42/eval.json --train 11.88=outputs/server/ctx-v1/ctx12_s42/eval.json \
        --control 11.88=outputs/server/ctx-v1/ctx12_b4_s42/eval.json \
        --own 2.96=outputs/server/ctx-v1/ctx3_s42/eval_crop130560.json --own 11.88=outputs/server/ctx-v1/ctx12_s42/eval_crop523776.json \
        --infer 2.96=outputs/server/ctx-v1/infer_loss-v1_baseline_s42/eval_crop130560.json --infer 11.88=...eval_crop523776.json
Outputs: figures/context_length.png, figures/context_length.json
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *

SHIFT = {"train": 1.0, "own": 1.03, "infer": 0.97, "control": 1.06}   # horizontal offset so that the error bars do not overlap
RF_SECONDS = 4.42   # measured total receptive field of the U-Net (381 STFT frames of 512 samples at 44.1 kHz), experiments/07_context_length.md


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
    for key, out in (("sdr", "vs_ref"), ("si_sdri", "vs_ref_si_sdri")):
        a = np.array([ref[t][key] for t in tr]); b = np.array([rows[t][key] for t in tr])
        d = b - a
        m, lo, hi = bootstrap_ci(d)
        r[out] = {"median_diff": m, "lo": lo, "hi": hi, "wins": int((d > 0).sum()),
                  "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0}
    return r


def main():
    ap = argparse.ArgumentParser()
    for name in ("train", "own", "infer", "control"):
        ap.add_argument(f"--{name}", action="append", default=[], metavar="SECONDS=eval.json")
    ap.add_argument("--rerun", action="append", default=[], metavar="NOM=eval.json", help="another training of the baseline")
    ap.add_argument("--ref", type=float, default=5.93, help="point of the --train series used as the baseline (default 5.93)")
    ap.add_argument("--out", default="context_length")
    args = ap.parse_args()
    if len(args.train) < 2:
        ap.error("at least two --train SECONDS=eval.json, including the baseline")
    setup_style()

    series = {n: pairs(getattr(args, n)) for n in ("train", "own", "infer", "control")}
    ref_path = dict(series["train"]).get(args.ref)
    if ref_path is None:
        ap.error(f"no --train {args.ref}=... (the baseline is the reference)")
    ref = load(ref_path)
    common = set(ref)
    reruns = dict(r.split("=", 1) for r in args.rerun)
    for s in series.values():
        for _, p in s:
            common &= set(load(p))
    for p in reruns.values():
        common &= set(load(p))
    res = {n: [row(sec, p, common, ref) for sec, p in s] for n, s in series.items()}
    res_rerun = {name: row(args.ref, p, common, ref) for name, p in reruns.items()}

    print(f"{len(common)} common test tracks; reference: baseline trained and evaluated on {args.ref} s")
    labels = {"train": "train L, eval 5.93 s", "own": "train L, eval L",
              "infer": "baseline, eval L", "control": "control"}
    for n, rows in res.items():
        if not rows:
            continue
        print(f"\n[{n}] {labels[n]}")
        print(f"  {'L (s)':>7s}  {'median SDR [95% CI]':24s}{'SI-SDRi':>9s}   vs baseline (paired SDR)")
        for r in rows:
            v = r["vs_ref"]; s = r["sdr"]
            print(f"  {r['seconds']:7.2f}  {s['median']:6.2f} [{s['lo']:.2f}, {s['hi']:.2f}]        {r['si_sdri']['median']:6.2f}   "
                  f"{v['median_diff']:+.2f} dB [{v['lo']:+.2f}, {v['hi']:+.2f}], better on {v['wins']}/{len(common)}, p={v['wilcoxon_p']:.2g}")
    for name, r in res_rerun.items():
        v = r["vs_ref"]
        print(f"  rerun {name:10s} median SDR {r['sdr']['median']:.2f}  vs baseline {v['median_diff']:+.2f} dB [{v['lo']:+.2f}, {v['hi']:+.2f}], "
              f"SI-SDRi {r['vs_ref_si_sdri']['median_diff']:+.2f} dB [{r['vs_ref_si_sdri']['lo']:+.2f}, {r['vs_ref_si_sdri']['hi']:+.2f}]")
    tr = res["train"]
    slope = slope_paired = None
    if len(tr) >= 3:
        x = np.log2([r["seconds"] for r in tr])
        slope = float(np.polyfit(x, [r["sdr"]["median"] for r in tr], 1)[0])
        slope_paired = float(np.polyfit(x, [r["vs_ref"]["median_diff"] for r in tr], 1)[0])
        print(f"\nSlope of the --train series: {slope:+.2f} dB per doubling on the medians, "
              f"{slope_paired:+.2f} dB on the paired differences (preferred)")

    labels = {"train": "train L, eval 5.93 s", "own": "train L, eval L",
              "infer": "baseline, eval L", "control": "control: 11.88 s, batch 4"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    style = {"train": (BLUE, "o", "-"), "own": (ORANGE, "s", "--"), "infer": (AQUA, "^", ":"), "control": (INK, "D", "none")}
    ticks = sorted({r["seconds"] for rows in res.values() for r in rows})
    panels = ((axes[0], "vs_ref", "a) SDR vs baseline (paired)", "Δ SDR (dB)"),
              (axes[1], "vs_ref_si_sdri", "b) SI-SDRi vs baseline (paired)", "Δ SI-SDRi (dB)"),
              (axes[2], None, "c) Test SDR (median)", "dB"))
    for ax, key, title, ylabel in panels:
        for n, rows in res.items():
            if not rows:
                continue
            c, mk, ls = style[n]
            xs = [r["seconds"] * SHIFT[n] for r in rows]
            if key:
                ys = [r[key]["median_diff"] for r in rows]
                err = [[y - r[key]["lo"] for y, r in zip(ys, rows)], [r[key]["hi"] - y for y, r in zip(ys, rows)]]
            else:
                ys = [r["sdr"]["median"] for r in rows]
                err = [[r["sdr"]["median"] - r["sdr"]["lo"] for r in rows], [r["sdr"]["hi"] - r["sdr"]["median"] for r in rows]]
            ax.errorbar(xs, ys, yerr=err, color=c, marker=mk, linestyle=ls, linewidth=2, markersize=7, capsize=3, label=labels[n], zorder=3)
            if key == "vs_ref":
                for x, y, r in zip(xs, ys, rows):
                    if r["seconds"] != args.ref or n != "train":
                        ax.annotate(f"{round(y, 2) + 0.0:+.2f}", (x, y), textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK_2)
        for i, (name, r) in enumerate(res_rerun.items()):  # retrainings of the baseline: hollow markers, slightly offset
            x = args.ref * (0.92 - 0.05 * i)
            y = r[key]["median_diff"] if key else r["sdr"]["median"]
            lo, hi = (r[key]["lo"], r[key]["hi"]) if key else (r["sdr"]["lo"], r["sdr"]["hi"])
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], color=BLUE, marker="o", markerfacecolor="white", markersize=7,
                        linestyle="none", capsize=3, zorder=4, label="baseline reruns" if i == 0 else None)
        if key:
            ax.axhline(0, color=INK_2, linestyle="--", linewidth=1)
        ax.axvline(RF_SECONDS, color=INK_2, linestyle=":", linewidth=1)
        ax.annotate(f"receptive field\n{RF_SECONDS} s", (RF_SECONDS, 0.98), xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", fontsize=8, color=INK_2, ha="right", va="top")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlabel("crop length L (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="lower center", fontsize=7)
    fig.suptitle(f"Context length, {len(common)} test tracks (one seed)",
                 fontweight="bold")
    save(fig, f"{args.out}.png")
    with open(f"{FIG_DIR}/{args.out}.json", "w") as f:
        json.dump({"tracks": len(common), "ref_seconds": args.ref, "slope_db_per_doubling": slope, "slope_paired_db_per_doubling": slope_paired,
                   "receptive_field_s": RF_SECONDS, **res, "reruns": res_rerun}, f, indent=2)
    print(f"-> figures/{args.out}.json")


if __name__ == "__main__":
    main()
