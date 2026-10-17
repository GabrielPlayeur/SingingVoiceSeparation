"""Distribution of the estimated mask on the 240 validation crops (phase 4 of docs/RERUN_PLAN.md), for two open questions:

- head bug (experiments/01, 11): with `legacy_head=True` the mask is sigmoid(sum_c w_c r_c + b) with r_c >= 0 (ReLU), so it is capped at
  sigmoid(b) only if all the head weights w_c of that output channel are <= 0. The old pre-fix checkpoint had a cap of 0.727 reached on
  46 % of the bins. Here: head weights and bias of the 1000-epoch legacy run, its maximum mask, and the share of bins at that maximum.
- compressed loss (experiments/05, 11): does |X|^0.3 in the loss push the mask to 0 where the voice is weak? Share of bins with a mask
  below 0.05 among the bins where the ideal ratio |V|/|X| (clipped to [0, 1]) is between 0.1 and 0.5, against the reference model.

Validation crops only (the same 240 crops as `val/si_sdr_median`), so no test track is looked at.
Usage: uv run python -m analysis.mask_stats --ckpt reference=PATH --ckpt legacy=PATH --ckpt power_loss=PATH
Outputs: figures/mask_stats.json, figures/mask_stats.png
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._common import *
from src.data import MUSDB18DataModule
from src.module import VoiceSeparationModule

BINS = np.linspace(0, 1, 101)
WEAK = (0.1, 0.5)   # ideal ratio range called "weak voice" (choice of this script)
ZERO = 0.05         # mask below this = "pushed to zero" (choice of this script)


def val_loader(ckpt, batch_size=4):
    hp = dict(torch.load(ckpt, map_location="cpu", weights_only=False)["datamodule_hyper_parameters"])
    hp.update(dataset_path=os.path.join(ROOT, "musdb18", "processed"), split_path=os.path.join(ROOT, "musdb18", "split.json"),
              num_workers=0)
    dm = MUSDB18DataModule(**hp)
    dm.setup("fit")
    return DataLoader(dm.val_dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def stats_of(ckpt, device):
    module = VoiceSeparationModule.load_from_checkpoint(ckpt, map_location="cpu").to(device).eval()
    hist = np.zeros(len(BINS) - 1)
    mmax, at_max, n_bins, weak, weak_zero = 0.0, 0, 0, 0, 0
    masks_first = []
    for batch in val_loader(ckpt):
        batch = module.on_after_batch_transfer({k: v.to(device) for k, v in batch.items()}, 0)
        mask = module(batch["mix_amp"])
        ideal = (batch["voc_amp"] / (batch["mix_amp"] + 1e-8)).clamp(0, 1)
        m = mask.flatten().cpu().numpy()
        hist += np.histogram(m, BINS)[0]
        masks_first.append(float(mask.max()))
        n_bins += m.size
        w = (ideal > WEAK[0]) & (ideal < WEAK[1])
        weak += int(w.sum())
        weak_zero += int((w & (mask < ZERO)).sum())
    # second pass for the share at the global maximum (the maximum is only known after the first pass)
    mmax = max(masks_first)
    for batch in val_loader(ckpt):
        batch = module.on_after_batch_transfer({k: v.to(device) for k, v in batch.items()}, 0)
        at_max += int((module(batch["mix_amp"]) >= mmax - 1e-6).sum())
    head = module.model.final[0]
    return dict(hist=hist.tolist(), mask_max=mmax, share_at_max=at_max / n_bins, n_bins=n_bins,
                weak_bins=weak, weak_zero_share=weak_zero / max(weak, 1), mean=float((hist * (BINS[:-1] + 0.005)).sum() / hist.sum()),
                head_weight=head.weight.squeeze().tolist(), head_bias=head.bias.tolist(),
                sigmoid_bias=torch.sigmoid(head.bias).tolist(), legacy_head=bool(module.hparams.get("legacy_head", False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    args = ap.parse_args()
    setup_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        r = stats_of(path, device)
        res[name] = r
        print(f"{name:12s} mask max {r['mask_max']:.3f}, share of bins at the max {r['share_at_max']:.1%}, mean mask {r['mean']:.3f}, "
              f"mask < {ZERO} on {r['weak_zero_share']:.1%} of the weak-voice bins ({r['weak_bins']} bins)")
        print(f"{'':12s} head weight {np.round(r['head_weight'], 3).tolist()}, bias {np.round(r['head_bias'], 3).tolist()}, "
              f"sigmoid(bias) {np.round(r['sigmoid_bias'], 3).tolist()}")
    with open(os.path.join(FIG_DIR, "mask_stats.json"), "w") as f:
        json.dump(dict(weak_range=WEAK, zero_threshold=ZERO, runs=res), f, indent=2)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for (name, r), c in zip(res.items(), (BLUE, ORANGE, AQUA, INK_2)):
        h = np.array(r["hist"]) / sum(r["hist"])
        ax.step(BINS[:-1], h, where="post", color=c, label=name)
    ax.set_yscale("log"); ax.set_xlabel("mask value"); ax.set_ylabel("share of time-frequency bins")
    ax.set_title("Mask distribution on the 240 validation crops"); ax.legend()
    save(fig, "mask_stats.png")


if __name__ == "__main__":
    main()
