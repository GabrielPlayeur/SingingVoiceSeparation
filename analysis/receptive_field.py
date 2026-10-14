"""Champ récepteur temporel du U-Net, mesuré par le gradient : combien de trames d'entrée influencent UNE trame de sortie ?

On rétropropage la somme du masque d'une trame centrale (tous canaux et toutes fréquences) vers l'entrée et on regarde où le
gradient est non nul. Deux largeurs :
  - totale : première à dernière trame de gradient non nul (le champ récepteur « théorique » : l'influence peut y être infime) ;
  - effective : trames dont le gradient dépasse `--threshold` x le maximum (seuil arbitraire : à lire comme un ordre de grandeur).
Poids aléatoires, mode eval (BatchNorm figée, dropout coupé) : le champ total ne dépend que de l'architecture ; le champ effectif
dépend aussi des poids, donc il est mesuré ici sur un réseau non entraîné (⚠ à refaire sur un checkpoint pour la valeur effective).
Pour chaque longueur d'extrait, donne la part des trames de sortie dont le contexte est tronqué par le bord de l'extrait (padding à zéro).

Usage : python -m analysis.receptive_field [--base-channels 16] [--depth 6] [--frames 2048] [--ckpt CKPT]
"""
import argparse

import torch

from src.models.unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--frames", type=int, default=2048, help="longueur de l'entrée de mesure (trames), multiple de 2**depth, > 2 x le champ")
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--ckpt", default=None, help="checkpoint dont on charge les poids (sinon poids aléatoires)")
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
    print(f"champ récepteur total    : {total} trames = {total * to_s:.2f} s")
    print(f"champ récepteur effectif : {eff} trames = {eff * to_s:.2f} s (gradient > {args.threshold:g} x max)")
    print("\nparts de trames de sortie dont le contexte est tronqué par le bord de l'extrait (padding à zéro) :")
    print(f"{'extrait':>10s} {'trames':>7s} {'champ total':>12s} {'champ effectif':>15s}")
    for crop in (130560, 261632, 392704, 523776):
        f = 1 + crop // args.hop
        print(f"{crop / args.sr:8.2f} s {f:7d} {min(1.0, (total - 1) / f):11.0%} {min(1.0, (eff - 1) / f):15.0%}")


if __name__ == "__main__":
    main()
