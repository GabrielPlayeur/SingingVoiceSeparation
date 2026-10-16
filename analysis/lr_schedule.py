"""Comparison of the baseline's optimisation runs (learning rate, scheduler) from the W&B history.

Selection criterion (on the VALIDATION set, never on the test set): mean of `val/si_sdr_median` over the LAST 10 epochs. The maximum
(used for the checkpoint) rewards a lucky epoch when the curve oscillates; the final mean measures what the run
actually delivers. The standard deviation over these 10 epochs gives the order of magnitude of the noise: below it, two runs cannot
be told apart (a single seed per run).

Usage: python -m analysis.lr_schedule --run stereo=fgvzjxua --run lr3e4=<W&B id> --run cosine=<id> --run lr3e4_cosine=<id>
The id is the one of the outputs/VoiceSeparation/<id>/ folder (or of the W&B run URL). Without --run: the baseline alone.
A server run that was never synced (offline W&B) is read directly from its file: --run lr3e4=path/run-<id>.wandb
(local reading of the binary W&B log, nothing is sent).
Outputs: figures/lr_schedule.png, figures/lr_schedule.json
"""
import argparse
import json
import os
import struct

import numpy as np
import matplotlib.pyplot as plt

from ._common import *

PROJECT = "playeur-audio-team/VoiceSeparation"
METRIC = "val/si_sdr_median"
LAST = 10  # number of epochs of the selection criterion
COLORS = [BLUE, ORANGE, AQUA, INK_2, "#8a5fd1"]


def _wandb_records(path):
    """Protobuf records of an offline W&B log (LevelDB format: 7-byte header, 32,768-byte blocks,
    each record preceded by crc(4) + length(2) + type(1); 1 = full, 2/3/4 = start/middle/end)."""
    data = open(path, "rb").read()
    block, pos, buf = 32768, 7, b""
    while pos < len(data):
        if block - pos % block < 7:
            pos += block - pos % block
            continue
        _, length, kind = struct.unpack("<IHB", data[pos:pos + 7])
        pos += 7
        if kind == 0 and length == 0:
            pos += (block - pos % block) % block
            continue
        chunk = data[pos:pos + length]
        pos += length
        if kind == 1:
            yield chunk
        elif kind == 2:
            buf = chunk
        elif kind == 3:
            buf += chunk
        elif kind == 4:
            yield buf + chunk
            buf = b""


def load_offline(path):
    from wandb.proto import wandb_internal_pb2 as pb
    rows = {}
    for raw in _wandb_records(path):
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") == "history":
            h = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in rec.history.item}
            if METRIC in h and "epoch" in h:
                rows[int(h["epoch"])] = h[METRIC]
    epochs = sorted(rows)
    return os.path.basename(path), np.array(epochs), np.array([rows[e] for e in epochs])


def load_history(run_id):
    if run_id.endswith(".wandb"):
        return load_offline(run_id)
    import wandb
    run = wandb.Api(timeout=30).run(f"{PROJECT}/{run_id}")
    h = run.history(keys=[METRIC, "epoch"], pandas=True, samples=5000).dropna(subset=[METRIC])
    h = h.drop_duplicates("epoch").sort_values("epoch")
    return run.name, h["epoch"].to_numpy(), h[METRIC].to_numpy()


def summarize(epochs, y):
    tail = y[-LAST:]
    return dict(epochs=int(len(y)), best=float(y.max()), best_epoch=int(epochs[y.argmax()]),
                last_mean=float(tail.mean()), last_std=float(tail.std(ddof=1)), final=float(y[-1]),
                epoch_to_epoch_std=float(np.diff(y).std(ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="NAME=WANDB_ID")
    args = ap.parse_args()
    setup_style()
    runs = dict(r.split("=", 1) for r in args.run) or {"stereo": "fgvzjxua"}

    results, curves = {}, {}
    for name, run_id in runs.items():
        wandb_name, epochs, y = load_history(run_id)
        results[name] = summarize(epochs, y) | {"wandb_run": run_id}
        curves[name] = (epochs, y)
        r = results[name]
        print(f"{name:14s} ({wandb_name}) {r['epochs']:3d} epochs | best {r['best']:.2f} dB (epoch {r['best_epoch']}) | "
              f"mean of the last {LAST} {r['last_mean']:.2f} ± {r['last_std']:.2f} dB | epoch-to-epoch noise {r['epoch_to_epoch_std']:.2f} dB")
    if len(results) > 1:
        ranked = sorted(results, key=lambda n: -results[n]["last_mean"])
        best = ranked[0]
        print(f"\nCriterion (mean of the last {LAST} epochs): " + " > ".join(f"{n} {results[n]['last_mean']:.2f}" for n in ranked))
        for n in ranked[1:]:
            gap = results[best]["last_mean"] - results[n]["last_mean"]
            noise = max(results[best]["last_std"], results[n]["last_std"])
            print(f"  {best} - {n}: {gap:.2f} dB (noise {noise:.2f} dB) -> {'gap above the noise' if gap > noise else 'not distinguishable'}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]})
    for i, (name, (epochs, y)) in enumerate(curves.items()):
        c = COLORS[i % len(COLORS)]
        ax.plot(epochs, y, color=c, alpha=0.25, linewidth=1)
        k = min(LAST, len(y))
        smooth = np.convolve(y, np.ones(k) / k, mode="valid")  # rolling mean over LAST epochs
        ax.plot(epochs[k - 1:], smooth, color=c, linewidth=2, label=f"{name} (fin : {results[name]['last_mean']:.2f} dB)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val/si_sdr_median (dB)")
    ax.set_title(f"Validation per epoch (thin line: raw, thick: rolling mean over {LAST})")
    ax.set_ylim(2.0, 4.3)  # the first epochs (from -4 to 2 dB) squash the scale; the comparison is played out above 2 dB
    ax.legend(loc="lower right")

    order = sorted(results, key=lambda n: results[n]["last_mean"])
    pos = np.arange(len(order))
    bx.barh(pos, [results[n]["last_mean"] for n in order], xerr=[results[n]["last_std"] for n in order],
            color=[COLORS[list(curves).index(n) % len(COLORS)] for n in order], error_kw={"ecolor": INK_2, "capsize": 3})
    for p_, n in zip(pos, order):
        bx.text(results[n]["last_mean"] + results[n]["last_std"] + 0.03, p_, f"{results[n]['last_mean']:.2f}", va="center", fontsize=9)
    bx.set_yticks(pos, order)
    bx.set_xlim(min(results[n]["last_mean"] for n in order) - 0.4, max(results[n]["last_mean"] for n in order) + 0.35)
    bx.set_xlabel(f"mean of the last {LAST} epochs ± standard deviation (dB)")
    bx.set_title("Selection criterion")
    bx.grid(axis="y", visible=False)
    save(fig, "lr_schedule.png")
    with open(os.path.join(FIG_DIR, "lr_schedule.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
