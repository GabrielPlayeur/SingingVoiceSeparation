"""Waveform-loss study (experiments/04): validation curves + paired test comparison against the baseline trained alongside.

Usage:
    python -m analysis.loss_study                                   # defaults: outputs/server/loss-v1, rerun = outputs/eval_stereo.json
    python -m analysis.loss_study --dir outputs/server/loss-v1 --runs baseline sisdr wsdr mrstft --rerun outputs/eval_stereo.json
    python -m analysis.loss_study --run baseline=outputs/server/loss-v1/baseline_s42 --run power_loss=outputs/server/power-v1/power_loss_s42 \
        --out power_study --title "Amplitude compression vs baseline"   # runs from different folders (the first = reference)
Outputs: figures/<out>.png, figures/<out>.json + text summary (default: loss_study).

If a run has no eval.json yet (evaluation not fetched), only the validation is plotted (curves + criterion of [03]).

Each run is a server folder `<dir>/<name>_s42/` (eval.json + offline W&B history). The reference of each paired
difference is the first run (the baseline trained on the same machine). `--rerun` adds a column "same
config, other machine" (the local baseline, same seed): this is the order of magnitude of the noise of a retraining, in the
absence of 3 seeds.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *
from .lr_schedule import LAST, load_offline, summarize

COLORS = {"baseline": INK_2, "rerun": "#a3a29e"}
SLOTS = [BLUE, ORANGE, AQUA, "#8a5fd1"]  # fixed order of the variants
CLIP = (-2.0, 2.0)  # scale of the differences; the points off the scale (PR tracks) are annotated


def load_eval(path):
    with open(path) as f:
        return {r["track"]: r for r in json.load(f)}


def paired(ref, other, metric, tracks):
    d = np.array([other[t][metric] - ref[t][metric] for t in tracks])
    m, lo, hi = bootstrap_ci(d)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0
    return dict(median=m, ci=[lo, hi], mean=float(d.mean()), wins=int((d > 0).sum()), n=len(d), p=p), d


def strip(ax, columns, metric, tracks):
    rng = np.random.default_rng(0)
    for x, (name, color, d, s) in enumerate(columns):
        jitter = rng.uniform(-0.18, 0.18, len(d))
        inside = (d >= CLIP[0]) & (d <= CLIP[1])
        ax.scatter(x + jitter[inside], d[inside], s=16, color=color, alpha=0.55, edgecolor="white", linewidth=0.5, zorder=3)
        for low in (True, False):  # off scale: one arrow at the edge, a single label per column and per edge
            idx = [i for i in np.where(~inside)[0] if (d[i] < CLIP[0]) == low]
            if not idx:
                continue
            y = CLIP[0] + 0.08 if low else CLIP[1] - 0.08
            ax.scatter([x] * len(idx), [y] * len(idx), marker="v" if low else "^", s=40, color=color, zorder=3)
            order = sorted(idx, key=lambda i: d[i] if low else -d[i])
            if len(order) <= 2:
                label = "\n".join(f"{tracks[i].split(' - ')[-1]}\n{d[i]:+.1f} dB" for i in order)
            else:  # too many off-scale tracks to name them: count + the worst
                label = f"{len(order)} tracks,\nworst:\n{tracks[order[0]].split(' - ')[-1]}\n{d[order[0]]:+.1f} dB"
            ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", fontsize=7, color=INK_2,
                        va="bottom" if low else "top")
        ax.plot([x - 0.3, x + 0.3], [s["median"]] * 2, color=INK, linewidth=2.5, zorder=4)
        ax.plot([x, x], s["ci"], color=INK, linewidth=1.2, zorder=4)
    ax.axhline(0, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(columns)), [f"{c[0]}\n{c[3]['median']:+.2f} dB\n[{c[3]['ci'][0]:+.2f}, {c[3]['ci'][1]:+.2f}]\n"
                                        f"{c[3]['wins']}/{c[3]['n']} better" for c in columns], fontsize=8)
    ax.set_xlim(-0.6, len(columns) - 0.2)
    ax.set_ylim(CLIP[0] - 0.1, CLIP[1] + 0.1)
    ax.set_ylabel(f"Δ {metric} vs baseline (dB)")
    ax.grid(axis="x", visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/server/loss-v1")
    ap.add_argument("--runs", nargs="+", default=["baseline", "sisdr", "wsdr", "mrstft"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=DIR", help="replaces --dir/--runs; the first one is the reference")
    ap.add_argument("--rerun", action="append", default=None, metavar="[NAME=]eval.json",
                    help="another training of the reference (same config, same seed), repeatable; '' to omit it "
                         "(default: 'baseline rerun'=outputs/eval_stereo.json)")
    ap.add_argument("--out", default="loss_study")
    ap.add_argument("--title", default="Waveform losses vs baseline")
    args = ap.parse_args()
    setup_style()
    reruns = [r for r in (args.rerun if args.rerun is not None else ["outputs/eval_stereo.json"]) if r]
    reruns = dict(r.split("=", 1) if "=" in r else ("baseline rerun", r) for r in reruns)
    dirs = dict(r.split("=", 1) for r in args.run) or {n: os.path.join(args.dir, f"{n}_s{args.seed}") for n in args.runs}
    args.runs = list(dirs)

    evals, curves, results = {}, {}, {}
    for name, run_dir in dirs.items():
        eval_path = os.path.join(run_dir, "eval.json")
        if os.path.exists(eval_path):
            evals[name] = load_eval(eval_path)
        wandb_file = glob.glob(os.path.join(run_dir, "outputs", "wandb", "offline-run-*", "run-*.wandb"))[0]
        _, epochs, y = load_offline(wandb_file)
        curves[name] = (epochs, y)
        results[name] = {"val": summarize(epochs, y)}
    ref_name = args.runs[0]
    color = {n: COLORS.get(n, SLOTS[i % len(SLOTS)]) for i, n in enumerate(n for n in args.runs if n != ref_name)}
    color[ref_name] = COLORS["baseline"]
    missing = [n for n in args.runs if n not in evals]
    if missing:
        print(f"No eval.json for {', '.join(missing)}: validation only.")
        print_val(args.runs, results)
        validation_only(args, curves, results, color)
        return
    ref = evals[ref_name]
    tracks = sorted(ref)
    no_pr = [t for t in tracks if not t.startswith("PR -")]

    others = [n for n in args.runs if n != ref_name]
    compare = {n: evals[n] for n in others}
    compare = {name: load_eval(path) for name, path in reruns.items()} | compare
    for name in reruns:
        color[name] = COLORS["rerun"]

    columns = {"sdr": [], "si_sdri": []}
    for name, ev in compare.items():
        results.setdefault(name, {})
        for metric in columns:
            s, d = paired(ref, ev, metric, tracks)
            s_no_pr, _ = paired(ref, ev, metric, no_pr)
            results[name][metric] = s | {"without_PR": s_no_pr}
            columns[metric].append((name, color[name], d, s))
    for name in [ref_name] + list(compare):
        ev = evals.get(name) or compare[name]
        results[name]["test_median"] = {m: bootstrap_ci([ev[t][m] for t in tracks]) for m in ("sdr", "si_sdri")}

    print_val(args.runs, results)
    print(f"\nTest, paired differences against {ref_name} ({len(tracks)} tracks; in parentheses: without the 2 PR tracks):")
    for name in compare:
        for metric in columns:
            s = results[name][metric]
            w = s["without_PR"]
            print(f"  {name:14s} {metric:7s} median {s['median']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] "
                  f"p = {s['p']:.2g}, {s['wins']}/{s['n']}, mean {s['mean']:+.2f}  "
                  f"({w['median']:+.2f} [{w['ci'][0]:+.2f}, {w['ci'][1]:+.2f}], p = {w['p']:.2g})")

    fig, axes = plt.subplots(1, 3, figsize=(14 + 1.4 * len(compare), 5.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    val_curves(axes[0], args.runs, curves, results, color)

    strip(axes[1], columns["sdr"], "SDR", tracks)
    axes[1].set_title("Test SDR (paired)")
    strip(axes[2], columns["si_sdri"], "SI-SDRi", tracks)
    axes[2].set_title("Test SI-SDRi (paired)")
    fig.suptitle(args.title, fontweight="bold")
    save(fig, f"{args.out}.png")
    with open(os.path.join(FIG_DIR, f"{args.out}.json"), "w") as f:
        json.dump(results, f, indent=2)


def print_val(runs, results):
    print(f"Validation (val/si_sdr_median, mean of the last {LAST} epochs ± standard deviation):")
    for name in runs:
        v = results[name]["val"]
        print(f"  {name:10s} {v['last_mean']:.2f} ± {v['last_std']:.2f} dB | best {v['best']:.2f} (ep. {v['best_epoch']}) | "
              f"epoch-to-epoch noise {v['epoch_to_epoch_std']:.2f} dB")


def val_curves(ax, runs, curves, results, color, min_gap=0.08):
    ends = {}
    for name in runs:
        epochs, y = curves[name]
        c = color[name]
        ax.plot(epochs, y, color=c, alpha=0.2, linewidth=1)
        smooth = np.convolve(y, np.ones(LAST) / LAST, mode="valid")
        ax.plot(epochs[LAST - 1:], smooth, color=c, linewidth=2, label=f"{name} ({results[name]['val']['last_mean']:.2f} dB)")
        ends[name] = (epochs[-1], smooth[-1])
    last = -np.inf  # end-of-curve labels: spaced at least min_gap dB apart so they do not overlap
    for name, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        last = max(y, last + min_gap)
        ax.text(x + 2, last, name, color=INK_2, fontsize=8, va="center")
    ax.set_xlim(0, max(e[-1] for e, _ in curves.values()) + 22)
    ax.set_ylim(1.5, 4.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val/si_sdr_median (dB)")
    ax.set_title(f"Validation ({LAST}-epoch mean)")
    ax.legend(loc="lower right", title=f"last-{LAST} mean", title_fontsize=8, fontsize=8)


def validation_only(args, curves, results, color):
    """Figure without a test: validation curves + selection criterion of [03] (mean of the LAST final epochs ± standard deviation)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    val_curves(ax, args.runs, curves, results, color)
    ref = results[args.runs[0]]["val"]
    order = list(args.runs)[::-1]
    pos = np.arange(len(order))
    means = [results[n]["val"]["last_mean"] for n in order]
    stds = [results[n]["val"]["last_std"] for n in order]
    bx.barh(pos, means, xerr=stds, color=[color[n] for n in order], height=0.6, error_kw={"ecolor": INK_2, "capsize": 3})
    for p_, n, m, sd in zip(pos, order, means, stds):
        delta = "" if n == args.runs[0] else f"  ({m - ref['last_mean']:+.2f})"
        bx.text(m + sd + 0.03, p_, f"{m:.2f}{delta}", va="center", fontsize=9)
    bx.axvline(ref["last_mean"], color=INK_2, linestyle="--", linewidth=1)
    bx.set_yticks(pos, order)
    bx.set_xlim(min(means) - 0.5, max(means) + 0.6)
    bx.set_xlabel(f"last-{LAST} mean ± std (dB)")
    bx.set_title("Validation")
    bx.grid(axis="y", visible=False)
    fig.suptitle(args.title, fontweight="bold")
    save(fig, f"{args.out}.png")
    with open(os.path.join(FIG_DIR, f"{args.out}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
