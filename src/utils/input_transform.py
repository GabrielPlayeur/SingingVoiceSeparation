"""Transformation de l'amplitude STFT avant le U-Net (ROADMAP, étape 2a, limite B10 du walkthrough).

Le masque reste appliqué à l'amplitude LINÉAIRE du mélange (`voc_est = masque * mix_amp`) : seule l'entrée du réseau change.

    none  : |X|                                  (baseline)
    log1p : log(1 + |X|)                         compresse les bins forts (> 1) ; quasi identité sous 1
    sqrt  : |X| ** 0.5                           compresse toute la dynamique (rapport 4 100 -> 64 entre médiane et max)
    norm  : |X| / moyenne(|X|) par exemple       invariant au gain exact ; la moyenne est calculée sur (canaux, fréquences, trames)
                                                 de l'extrait de 6 s, donc aussi à l'inférence (par fenêtre, sans état)

Aucune n'est « invariante au volume » sauf `norm` : voir analysis/input_amplitude.py pour la mesure.
"""
import torch

INPUT_TRANSFORMS = ("none", "log1p", "sqrt", "norm")


def transform_amplitude(amp: torch.Tensor, kind: str) -> torch.Tensor:
    """amp : (B, C, F, T) amplitudes >= 0 -> même forme."""
    if kind == "none":
        return amp
    if kind == "log1p":
        return torch.log1p(amp)
    if kind == "sqrt":
        return torch.sqrt(amp)
    if kind == "norm":
        return amp / amp.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)  # clamp : extrait entièrement silencieux
    raise ValueError(f"input_transform={kind!r} inconnu, choisir parmi {INPUT_TRANSFORMS}")
