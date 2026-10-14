"""Transformation de l'amplitude STFT avant le U-Net (ROADMAP, étape 2a, limite B10 du walkthrough).

Le masque reste appliqué à l'amplitude LINÉAIRE du mélange (`voc_est = masque * mix_amp`) : seule l'entrée du réseau change.

    none  : |X|                                  (baseline)
    power : |X| ** c                             loi de puissance, c = 0,3 par défaut : exposant de Wilson et al. 2018 et de
                                                 Braun & Tashev 2020 (speech enhancement, à confirmer sur la musique)
    log1p : log(1 + |X|)                         compresse les bins forts (> 1) ; quasi identité sous 1 (87 % des bins sont < 1)
    sqrt  : |X| ** 0.5                           loi de puissance c = 0,5
    norm  : |X| / moyenne(|X|) par exemple       invariant au gain exact ; la moyenne est calculée sur (canaux, fréquences, trames)
                                                 de l'extrait de 6 s, donc aussi à l'inférence (par fenêtre, sans état)

La même loi de puissance peut aussi être appliquée dans la loss (`compress`) : c'est l'objet de Braun & Tashev.

Aucune n'est « invariante au volume » sauf `norm` : voir analysis/input_amplitude.py pour la mesure.
"""
import torch

INPUT_TRANSFORMS = ("none", "power", "log1p", "sqrt", "norm")
EPS = 1e-8  # plancher de la loi de puissance : x ** c a une dérivée infinie en 0 pour c < 1


def compress(amp: torch.Tensor, power: float) -> torch.Tensor:
    """Loi de puissance (amp + EPS) ** power ; power = 1 rend amp inchangé (baseline exacte)."""
    return amp if power == 1.0 else (amp + EPS) ** power


def transform_amplitude(amp: torch.Tensor, kind: str, power: float = 0.3) -> torch.Tensor:
    """amp : (B, C, F, T) amplitudes >= 0 -> même forme. `power` ne sert que pour kind == "power"."""
    if kind == "none":
        return amp
    if kind == "power":
        return compress(amp, power)
    if kind == "log1p":
        return torch.log1p(amp)
    if kind == "sqrt":
        return torch.sqrt(amp)
    if kind == "norm":
        return amp / amp.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)  # clamp : extrait entièrement silencieux
    raise ValueError(f"input_transform={kind!r} inconnu, choisir parmi {INPUT_TRANSFORMS}")
