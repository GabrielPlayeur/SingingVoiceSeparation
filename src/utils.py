"""
Usage :
    python -m src.utils SR HOP DEPTH TARGET_SECONDS
"""

import argparse

import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SI-SDR de signaux (B, C, T) ou (C, T), moyenné sur les canaux -> (B,) ou scalaire."""
    return scale_invariant_signal_distortion_ratio(estimate, target).mean(dim=-1)

def compute_crop_length(sr: int, hop_length: int, depth: int, target_seconds: float) -> int:
    """Longueur de crop (en échantillons) la plus proche de target_seconds telle que le
    nombre de trames STFT soit un multiple de 2**depth : aucun padding temporel dans le
    U-Net."""
    m = 2 ** depth
    target_frames = 1 + int(sr * target_seconds) // hop_length
    k = max(round(target_frames / m), 1)
    return (k * m - 1) * hop_length


def assert_no_temporal_padding(n_samples: int, hop_length: int, depth: int) -> None:
    frames = 1 + n_samples // hop_length
    m = 2 ** depth
    if frames % m != 0:
        raise ValueError(
            f"crop_length={n_samples} echantillons donne {frames} trames STFT, "
            f"non multiple de {m} (2**depth, depth={depth}) : "
            f"le U-Net va ajouter du padding temporel (voir B13)."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sr")
    parser.add_argument("hop_length")
    parser.add_argument("depth")
    parser.add_argument("target_seconds")
    args = parser.parse_args()
    crop_length = compute_crop_length(args.sr, args.hop_length, args.depth, args.target_seconds)
    print(f"CROP LENGTH: {crop_length}")