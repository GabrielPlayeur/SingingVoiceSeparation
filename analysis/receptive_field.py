"""Temporal receptive field of the U-Net, measured by gradient: how many input frames influence ONE output frame?

The mask sum of a central frame (all channels and all frequencies) is backpropagated to the input and we look at where the
gradient is non-zero. Two widths:
  - total: first to last frame with non-zero gradient (the "theoretical" receptive field: the influence can be tiny there);
  - effective: frames whose gradient exceeds `--threshold` x the maximum (arbitrary threshold: to be read as an order of magnitude).
Random weights, eval mode (BatchNorm frozen, dropout off): the total field only depends on the architecture; the effective field
also depends on the weights, so it is measured here on an untrained network (⚠ to be redone on a checkpoint for the effective value).
For each crop length, gives the share of output frames whose context is truncated by the crop edge (zero padding).

Usage: python -m analysis.receptive_field [--base-channels 16] [--depth 6] [--frames 2048] [--ckpt CKPT]
"""
import argparse

import torch

from src.models.unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--frames", type=int, default=2048, help="length of the measurement input (frames), multiple of 2**depth, > 2 x the field")
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--ckpt", default=None, help="checkpoint whose weights are loaded (otherwise random weights)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.ckpt:
        from src.module import VoiceSeparationModule
        model = VoiceSeparationModule.load_from_checkpoint(args.ckpt, map_location="cpu").model
    else:
        model = UNet(args.base_channels, args.depth, args.in_channels)
    model.eval()
    x = torch.rand(1, args.in_channels, 1025, args.frames, requires_grad=True)
    c = args.frames // 2
    model(x)[0, :, :, c].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1, 2))
    to_s = args.hop / args.sr

    def width(mask):
        nz = mask.nonzero().flatten()
        return int(nz.max() - nz.min() + 1)

    total, eff = width(g > 0), width(g > g.max() * args.threshold)
    print(f"total receptive field    : {total} frames = {total * to_s:.2f} s")
    print(f"effective receptive field: {eff} frames = {eff * to_s:.2f} s (gradient > {args.threshold:g} x max)")
    print("\nshares of output frames whose context is truncated by the crop edge (zero padding):")
    print(f"{'crop':>10s} {'frames':>7s} {'champ total':>12s} {'champ effectif':>15s}")
    for crop in (130560, 261632, 392704, 523776):
        f = 1 + crop // args.hop
        print(f"{crop / args.sr:8.2f} s {f:7d} {min(1.0, (total - 1) / f):11.0%} {min(1.0, (eff - 1) / f):15.0%}")


if __name__ == "__main__":
    main()
