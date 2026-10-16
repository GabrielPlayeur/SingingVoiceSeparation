"""Level of the estimated vocals: does the model output the right scale? (ROADMAP 2b, Le Roux et al. 2019 § 2.3)

SI-SDR does not penalise a gain error (it rescales the target to the optimal scale, see `scale_error`),
whereas the evaluation SDR (museval) and the SNR depend on it. A -SI-SDR loss therefore gives no signal on the output
level. Since the mask is bounded in [0, 1], the possible drift is vocals that are too quiet (α > 1).

This script measures, per validation crop and for each checkpoint:
  - SI-SDR and SNR (dB) of the estimated vocals;
  - the optimal gain α (in dB; > 0: estimated vocals too quiet, < 0: too loud);
  - `scale cost` = SNR(α·ŝ) - SNR(ŝ) >= 0 (dB): what the SNR would gain if the estimated vocals had the right level;
    0 = perfect level. (Careful: SI-SDR - SNR is NOT this cost; SI-SDR is not bounded by the SNR, see Le Roux § 2.2.)

Usage: python -m analysis.output_scale [--ckpt NAME=PATH ...] [--crops-per-track 5]
Without --ckpt: the stereo baseline (run fgvzjxua). Outputs: figures/output_scale.png, figures/output_scale.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from ._common import *
from .input_amplitude import DEFAULT_CKPTS, load_val_crops
from src.module import VoiceSeparationModule
from src.utils.utils import si_sdr


def scale_error(est, target):
    """est, target (B, C, T) -> SI-SDR, SNR, optimal gain α (dB) and scale cost (dB), averaged over the channels.

    SNR = 10 log10(‖s‖² / ‖s - ŝ‖²) is the scale taken "as is" (Le Roux et al., eq. 1);
    α = <ŝ, s> / <ŝ, ŝ>: factor by which ŝ must be multiplied to align with s."""
    snr = lambda e: 10 * torch.log10(target.pow(2).sum(-1) / (target - e).pow(2).sum(-1).clamp_min(1e-12))
    alpha = ((est * target).sum(-1) / est.pow(2).sum(-1).clamp_min(1e-12)).clamp_min(1e-6)  # (B, C)
    cost = snr(alpha[..., None] * est) - snr(est)
    return si_sdr(est, target), snr(est).mean(-1), (20 * torch.log10(alpha)).mean(-1), cost.mean(-1)


@torch.no_grad()
def measure(module, mixture, vocals, device, batch_size=8):
    module = module.to(device).eval()
    out = {"si_sdr": [], "snr": [], "alpha_db": [], "cost": []}
    for i in range(0, len(mixture), batch_size):
        mix, voc = mixture[i:i + batch_size].to(device), vocals[i:i + batch_size].to(device)
        if module.hparams.in_channels == 1:  # old mono model: applied to L then R
            est = torch.cat([module.predict_step(module.on_after_batch_transfer({"mixture": mix[:, c:c + 1]}, 0), 0) for c in range(2)], 1)
        else:
            est = module.predict_step(module.on_after_batch_transfer({"mixture": mix}, 0), 0)
        for k, v in zip(out, scale_error(est, voc)):
            out[k].append(v.cpu())
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--crops-per-track", type=int, default=5)
    args = ap.parse_args()
    setup_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    if not ckpts:
        for name, run in DEFAULT_CKPTS.items():
            found = [c for c in glob.glob(os.path.join(ROOT, f"outputs/VoiceSeparation/{run}/checkpoints/*.ckpt"))
                     if not os.path.basename(c).startswith(("last", "periodic"))]
            if found:
                ckpts[name] = found[0]

    mixture, vocals = load_val_crops(args.crops_per_track)
    print(f"{len(mixture)} validation crops")
    results = {}
    for name, path in ckpts.items():
        r = measure(VoiceSeparationModule.load_from_checkpoint(path, map_location="cpu"), mixture, vocals, device)
        results[name] = r
        print(f"{name:14s} SI-SDR {np.median(r['si_sdr']):5.2f} dB | SNR {np.median(r['snr']):5.2f} dB | median scale cost "
              f"{np.median(r['cost']):.2f} dB (p90 {np.percentile(r['cost'], 90):.2f}) | median optimal gain {np.median(r['alpha_db']):+.2f} dB "
              f"(p10 {np.percentile(r['alpha_db'], 10):+.2f}, p90 {np.percentile(r['alpha_db'], 90):+.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    colors = [BLUE, ORANGE, AQUA]
    for i, (name, r) in enumerate(results.items()):
        c = colors[i % len(colors)]
        axes[0].hist(r["cost"], bins=np.linspace(0, max(3, np.percentile(r["cost"], 99)), 40), alpha=0.55, color=c, label=name)
        axes[1].hist(r["alpha_db"], bins=np.linspace(-10, 10, 41), alpha=0.55, color=c, label=name)
    axes[0].set_xlabel("scale cost: SNR(α·ŝ) − SNR(ŝ)  (dB)")
    axes[0].set_ylabel("validation crops")
    axes[0].set_title("What the output level costs")
    axes[1].axvline(0, color=INK_2, linewidth=1, linestyle="--")
    axes[1].set_xlabel("optimal gain α of the estimated vocals (dB)   [> 0: estimated too quiet]")
    axes[1].set_title("Level of the estimated vocals")
    axes[0].legend()
    save(fig, "output_scale.png")
    with open(os.path.join(FIG_DIR, "output_scale.json"), "w") as f:
        json.dump({n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in results.items()}, f)


if __name__ == "__main__":
    main()
