"""Convergence probe (phase 1) and recipe decision (phase 2) of the re-run plan (docs/RERUN_PLAN.md §4-5).

The decision rules are CONSTANTS of this file, written before any probe number exists: they are not command-line options, so they
cannot be tuned after looking at the curves. All inputs are offline W&B logs (`run-<id>.wandb`, read locally with the reader of
lr_schedule.py) or a run directory containing one. Epoch convention: "epoch E" = after E completed epochs, i.e. the 0-indexed
epochs E-10 ... E-1; it is the content of the periodic checkpoint `periodic-epoch{E-1}-step...ckpt`.

Subcommands
  budget --run w16_aug=PATH --run w32_aug=PATH
      Rule of B (RERUN_PLAN §4): validation (`val/si_sdr_median`) smoothed by a trailing mean over SMOOTH=20 epochs; B is the first
      epoch at which the smoothed gain over HORIZON=100 epochs (value 100 epochs later minus value now) falls below TOL=0.05 dB,
      rounded UP to the next multiple of 100 (to land on a periodic checkpoint). B of the study = the largest B of the runs given (B must hold for the
      biggest model compared). Curve still rising at the end -> B = run length. The first crossing is used, not "stays below until the
      end": the epoch-to-epoch noise (0.18 dB std of the differences on a 150-epoch baseline, measured) makes the gain of a 20-epoch
      mean fluctuate by about 0.04 dB, so any later excursion above 0.05 dB would push B to ~900 whatever the curve (simulation on
      4*(1-exp(-e/100)) + noise: persistence variant 800-900, first crossing 400, noise-free 500). The max gain after the crossing is
      printed as a check.
  at --epoch E --run NAME=PATH ...
      Mean +/- std of the last LAST=10 validation epochs at epoch E (the selection criterion of page 03, at any epoch of a run).
  plot --run NAME=PATH ... [--budget B]
      Validation, training loss and validation loss, and their gap -> figures/convergence.png (overfitting check).
  decide --budget B --cell const=P,P,P --cell const_aug=... --cell wsd=... --cell wsd_aug=...
      Phase 2 rule: sigma_seed pooled over the 4 cells, threshold 2 * sigma_seed * sqrt(2/3), factors kept.
Usage: uv run python -m analysis.convergence budget --run w16_aug=outputs/server/probe-v1/probe_w16_aug_s42 --run w32_aug=...
Outputs: figures/convergence*.json (+ figures/convergence.png for `plot`).
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

from ._common import *
from .lr_schedule import _wandb_records

VAL = "val/si_sdr_median"
TRAIN_LOSS, VAL_LOSS = "train/loss", "val/loss"
LAST = 10      # epochs of the selection criterion (page 03)
SMOOTH = 20    # trailing mean for the budget rule
HORIZON = 100  # epochs over which the gain is measured
TOL = 0.05     # dB per HORIZON epochs: a third of the +/-0.15 dB re-run noise (experiments/06), see RERUN_PLAN §4
STEP = 100     # B is rounded up to a multiple of STEP (periodic checkpoint every 100 epochs)
CELLS = ("const", "const_aug", "wsd", "wsd_aug")
COLORS = [BLUE, ORANGE, AQUA, INK_2, "#8a5fd1"]


def find_wandb(path):
    """A `.wandb` file, or a run directory (<run>/outputs/wandb/offline-run-*/run-*.wandb, or the wandb folder itself)."""
    if os.path.isfile(path):
        return path
    found = sorted(glob.glob(os.path.join(path, "**", "run-*.wandb"), recursive=True))
    if len(found) != 1:
        sys.exit(f"{path}: expected exactly one run-*.wandb, found {len(found)}")
    return found[0]


def read_run(path, keys=(VAL, TRAIN_LOSS, VAL_LOSS)):
    """{key: array indexed by epoch (NaN where the key was not logged)} from an offline W&B log."""
    from wandb.proto import wandb_internal_pb2 as pb
    rows = {k: {} for k in keys}
    for raw in _wandb_records(find_wandb(path)):
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        h = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in rec.history.item}
        if "epoch" in h:
            for k in keys:
                if k in h:
                    rows[k][int(h["epoch"])] = h[k]
    out = {}
    for k, d in rows.items():
        if d:
            y = np.full(max(d) + 1, np.nan)
            y[list(d)] = list(d.values())
            out[k] = y
    return out


def window_mean(y, epoch, last=LAST):
    """Mean and std (ddof=1) of the `last` epochs ending at `epoch` (= after `epoch` completed epochs)."""
    if epoch > len(y) or epoch < last:
        raise ValueError(f"epoch {epoch} outside the run ({len(y)} epochs logged)")
    w = y[epoch - last:epoch]
    if np.isnan(w).any():
        raise ValueError(f"missing validation epochs in [{epoch - last}, {epoch - 1}]")
    return float(w.mean()), float(w.std(ddof=1))


def budget_epoch(y, smooth=SMOOTH, horizon=HORIZON, tol=TOL, step=STEP):
    """Rule of B on one validation curve. Returns dict(B, epoch0, rising, ...). Parameters are arguments for tests only."""
    if np.isnan(y).any():
        raise ValueError("gaps in the validation curve")
    s = np.convolve(y, np.ones(smooth) / smooth, mode="valid")   # s[i] = mean of epochs i .. i+smooth-1 = value at epoch i+smooth-1
    gain = s[horizon:] - s[:-horizon]                              # gain[i] = s(e + horizon) - s(e), e = i + smooth - 1
    below = gain < tol
    n = len(y)
    if not below.any():
        return dict(B=n, epoch0=None, rising=True, max_gain_after=None)
    i = int(np.argmax(below))                                      # first crossing, as the plan's wording says
    e0 = i + smooth - 1                                            # 0-indexed epoch; e0 + 1 epochs completed
    return dict(B=min(n, int(math.ceil((e0 + 1) / step) * step)), epoch0=int(e0), rising=False, max_gain_after=float(gain[i:].max()))


def pooled_sigma(cells):
    """sigma_seed pooled over the cells (equal n): sqrt of the mean within-cell variance (ddof=1). Returns (sigma, dof)."""
    var = [np.var(v, ddof=1) for v in cells.values()]
    return float(np.sqrt(np.mean(var))), sum(len(v) - 1 for v in cells.values())


def decide(cells):
    """cells: {const, const_aug, wsd, wsd_aug: per-seed scores}. Rule of RERUN_PLAN §5 (3 seeds per cell).

    A factor is kept if its main effect (mean of its two simple effects, i.e. 6 runs against 6) exceeds 2 * sigma * sqrt(2/3), the
    plan's threshold for a difference of two 3-seed means. For a 6-vs-6 contrast the true SD is sigma * sqrt(1/3), so this
    threshold is conservative (reported as `thr_exact_2sd`, information only). A simple effect of opposite sign is flagged.
    """
    n = {k: len(v) for k, v in cells.items()}
    if set(n) != set(CELLS) or len(set(n.values())) != 1:
        raise ValueError(f"expected the 4 cells {CELLS} with the same number of seeds, got {n}")
    m = {k: float(np.mean(v)) for k, v in cells.items()}
    sigma, dof = pooled_sigma(cells)
    k = n["const"]
    thr = 2 * sigma * math.sqrt(2 / k)
    out = dict(sigma_seed=sigma, dof=dof, seeds_per_cell=k, threshold=thr, means=m, factors={})
    for name, (a_off, a_on, b_off, b_on) in {"augmentation": ("const", "const_aug", "wsd", "wsd_aug"),
                                              "lr_decay (wsd)": ("const", "wsd", "const_aug", "wsd_aug")}.items():
        e1, e2 = m[a_on] - m[a_off], m[b_on] - m[b_off]
        main = (e1 + e2) / 2
        out["factors"][name] = dict(simple_effects=[e1, e2], main_effect=main, kept=bool(main > thr),
                                    sign_disagreement=bool(e1 * e2 < 0), thr_exact_2sd=2 * sigma * math.sqrt(1 / k))
    inter = (m["wsd_aug"] - m["wsd"]) - (m["const_aug"] - m["const"])
    # SD of the interaction contrast (+1,-1,-1,+1 over 4 cell means) = sigma * sqrt(4/k)
    out["interaction"] = dict(value=inter, thr_2sd=2 * sigma * math.sqrt(4 / k))
    aug, wsd = out["factors"]["augmentation"]["kept"], out["factors"]["lr_decay (wsd)"]["kept"]
    out["recipe"] = " ".join(["experiment=probe_w16_aug" if aug else "experiment=probe_w16"] + (["model.scheduler=wsd"] if wsd else []))
    return out


def parse_runs(specs):
    runs = {}
    for s in specs:
        name, _, path = s.partition("=")
        if not path:
            sys.exit(f"--run expects NAME=PATH, got {s!r}")
        runs[name] = path
    return runs


def cmd_budget(args):
    res = {}
    for name, path in parse_runs(args.run).items():
        r = budget_epoch(read_run(path, (VAL,))[VAL])
        res[name] = r
        if r["rising"]:
            what = "gain never below the tolerance: still rising at the end of the run"
        else:
            what = (f"smoothed gain < {TOL} dB/{HORIZON} ep from epoch {r['epoch0'] + 1}; "
                    f"largest gain afterwards {r['max_gain_after']:+.3f} dB/{HORIZON} ep")
        print(f"{name:10s} B = {r['B']:4d}  ({what})")
    B = max(r["B"] for r in res.values())
    print(f"\nB = {B} (largest of the runs given)")
    save_json("convergence_budget.json", dict(B=B, runs=res, rule=dict(smooth=SMOOTH, horizon=HORIZON, tol=TOL, step=STEP)))


def cmd_at(args):
    res = {}
    for name, path in parse_runs(args.run).items():
        m, s = window_mean(read_run(path, (VAL,))[VAL], args.epoch)
        res[name] = dict(mean=m, std=s)
        print(f"{name:14s} epoch {args.epoch}: mean of the last {LAST} = {m:.3f} +/- {s:.3f} dB")
    save_json("convergence_at.json", dict(epoch=args.epoch, runs=res))


def cmd_plot(args):
    setup_style()
    runs = {n: read_run(p) for n, p in parse_runs(args.run).items()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    for i, (name, d) in enumerate(runs.items()):
        c = COLORS[i % len(COLORS)]
        v = d[VAL]
        x = np.arange(len(v))
        axes[0].plot(x, v, color=c, alpha=0.2, linewidth=1)
        axes[0].plot(x[SMOOTH - 1:], np.convolve(v, np.ones(SMOOTH) / SMOOTH, mode="valid"), color=c, linewidth=2, label=name)
        if TRAIN_LOSS in d and VAL_LOSS in d:
            n = min(len(d[TRAIN_LOSS]), len(d[VAL_LOSS]))
            axes[1].plot(np.arange(n), d[TRAIN_LOSS][:n], color=c, linewidth=1.5, label=f"{name} train")
            axes[1].plot(np.arange(n), d[VAL_LOSS][:n], color=c, linewidth=1.5, linestyle="--", label=f"{name} val")
            axes[2].plot(np.arange(n), d[VAL_LOSS][:n] - d[TRAIN_LOSS][:n], color=c, linewidth=1.5, label=name)
    for ax in axes:
        ax.set_xlabel("epoch")
        if args.budget:
            ax.axvline(args.budget, color=INK_2, linestyle=":", linewidth=1)
    axes[0].set_ylabel("val/si_sdr_median (dB)")
    axes[0].set_title(f"Validation (thin: raw, thick: trailing mean over {SMOOTH})")
    axes[0].set_ylim(bottom=2.0)
    axes[1].set_ylabel("loss"); axes[1].set_title("Training and validation loss")
    axes[2].set_ylabel("val loss - train loss"); axes[2].set_title("Gap (overfitting if it grows)")
    axes[2].axhline(0, color=INK_2, linewidth=1)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Training loss of the augmented runs is measured on augmented data: compare gaps within a run, not between "
                 "augmented and plain runs", fontsize=9, color=INK_2)
    save(fig, "convergence.png")


def cmd_decide(args):
    cells = {}
    for spec in args.cell:
        name, _, paths = spec.partition("=")
        cells[name] = [window_mean(read_run(p, (VAL,))[VAL], args.budget)[0] for p in paths.split(",")]
    r = decide(cells)
    print(f"Budget B = {args.budget} | score = mean of val/si_sdr_median over the last {LAST} epochs")
    for k in CELLS:
        print(f"  {k:10s} " + "  ".join(f"{x:.3f}" for x in cells[k]) + f"   mean {r['means'][k]:.3f}")
    print(f"sigma_seed = {r['sigma_seed']:.3f} dB (pooled, {r['dof']} dof) | threshold 2*sigma*sqrt(2/{r['seeds_per_cell']}) = {r['threshold']:.3f} dB")
    for name, f in r["factors"].items():
        warn = "  [!] simple effects of opposite sign: decide per cell" if f["sign_disagreement"] else ""
        print(f"  {name:15s} main effect {f['main_effect']:+.3f} dB (simple: {f['simple_effects'][0]:+.3f}, {f['simple_effects'][1]:+.3f})"
              f" -> {'KEPT' if f['kept'] else 'not kept'}{warn}")
    i = r["interaction"]
    print(f"  interaction     {i['value']:+.3f} dB (2 SD = {i['thr_2sd']:.3f} dB; information only)")
    print(f"RECIPE: {r['recipe']}")
    save_json("convergence_decide.json", dict(budget=args.budget, cells=cells, **r))


def save_json(name, obj):
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"-> figures/{name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("budget", "at", "plot"):
        p = sub.add_parser(name)
        p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
        if name == "at":
            p.add_argument("--epoch", type=int, required=True, help="after this many completed epochs")
        if name == "plot":
            p.add_argument("--budget", type=int, default=None)
    p = sub.add_parser("decide")
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--cell", action="append", required=True, metavar="NAME=PATH,PATH,PATH")
    args = ap.parse_args()
    {"budget": cmd_budget, "at": cmd_at, "plot": cmd_plot, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
