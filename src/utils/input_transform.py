"""Transformation of the STFT amplitude before the U-Net (ROADMAP, step 2a, walkthrough limit B10).

The mask is still applied to the LINEAR amplitude of the mixture (`voc_est = mask * mix_amp`): only the network input changes.

    none  : |X|                                  (baseline)
    power : |X| ** c                             power law, c = 0.3 by default: exponent of Wilson et al. 2018 and of
                                                 Braun & Tashev 2020 (speech enhancement, to be confirmed on music)
    log1p : log(1 + |X|)                         compresses the strong bins (> 1); almost the identity below 1 (87 % of the bins are < 1)
    sqrt  : |X| ** 0.5                           power law c = 0.5
    norm  : |X| / mean(|X|) per example          invariant to an exact gain; the mean is taken over (channels, frequencies, frames)
                                                 of the 6 s crop, hence also at inference (per window, stateless)

The same power law can also be applied in the loss (`compress`): this is the subject of Braun & Tashev.

None is "volume invariant" except `norm`: see analysis/input_amplitude.py for the measurement.
"""
import torch

INPUT_TRANSFORMS = ("none", "power", "log1p", "sqrt", "norm")
EPS = 1e-8  # floor of the power law: x ** c has an infinite derivative at 0 for c < 1


def compress(amp: torch.Tensor, power: float) -> torch.Tensor:
    """Power law (amp + EPS) ** power; power = 1 returns amp unchanged (exact baseline)."""
    return amp if power == 1.0 else (amp + EPS) ** power


def transform_amplitude(amp: torch.Tensor, kind: str, power: float = 0.3) -> torch.Tensor:
    """amp: (B, C, F, T) amplitudes >= 0 -> same shape. `power` is only used when kind == "power"."""
    if kind == "none":
        return amp
    if kind == "power":
        return compress(amp, power)
    if kind == "log1p":
        return torch.log1p(amp)
    if kind == "sqrt":
        return torch.sqrt(amp)
    if kind == "norm":
        return amp / amp.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)  # clamp: fully silent crop
    raise ValueError(f"unknown input_transform={kind!r}, choose among {INPUT_TRANSFORMS}")
