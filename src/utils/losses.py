"""Losses calculées sur la FORME D'ONDE estimée (ROADMAP, étape 2b, limite e du walkthrough).

La forme d'onde est `istft(masque * |X|, phase du mélange)` (`VoiceSeparationModule._to_wave`, différentiable) : la loss voit donc
la reconstruction réelle, phase du mélange comprise, c'est-à-dire ce que l'évaluation mesure. Toutes prennent des tenseurs
(B, C, T) et renvoient un scalaire à MINIMISER.

    sisdr   : -SI-SDR (dB)                                        Le Roux et al. 2019, eq. 4-5
    wsdr    : -[α·cos(y, ŷ) + (1-α)·cos(z, ẑ)], z = x - y         Choi et al. 2019 (ICLR), eq. 7 et 9
    mrstft  : moyenne sur M résolutions de (L_sc + L_mag)         Yamamoto et al. 2020, eq. 3-6 et table 1

Rien ici n'est tiré d'une bibliothèque : les formules sont celles des papiers, relues en entier (voir README).
"""
import torch
from torch.utils.checkpoint import checkpoint

from .utils import si_sdr

EPS = 1e-8  # dénominateurs de wsdr : « we add small number ε to denominators » (Choi et al., annexe C) ; valeur choisie ici

# Table 1 de Yamamoto et al. (24 kHz) : (FFT, fenêtre, pas) = (1024, 600, 120), (2048, 1200, 240), (512, 240, 50), soit des
# fenêtres de 25 / 50 / 10 ms et des pas de 5 / 10 / ~2 ms. Ramenées à 44,1 kHz EN CONSERVANT LES DURÉES (⚠ c : adaptation faite
# ici, le papier ne traite que du 24 kHz), FFT = puissance de 2 qui contient la fenêtre.
MRSTFT_RESOLUTIONS = (
    (2048, 1102, 220),  # fenêtre 25 ms, pas 5 ms
    (4096, 2205, 441),  # fenêtre 50 ms, pas 10 ms
    (1024, 441, 92),    # fenêtre 10 ms, pas ~2 ms
)
MRSTFT_FLOOR = 1e-7  # plancher de |STFT|² : d|z|/dz n'est pas défini en 0 ; choix d'implémentation, absent du papier

LOSS_TYPES = ("amp_l1", "sisdr", "wsdr", "mrstft")


def sisdr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-SI-SDR moyen sur (exemples, canaux). Même fonction que la métrique de validation (`utils.si_sdr`), donc même formule
    que `val/si_sdr_*` : on optimise exactement ce qui est suivi. Invariante à l'échelle de `est` (Le Roux et al., § 2.3)."""
    return -si_sdr(est, target).mean()


def wsdr_loss(mixture: torch.Tensor, target: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
    """Weighted-SDR de Choi et al. : α·loss_SDR(y, ŷ) + (1-α)·loss_SDR(z, ẑ), loss_SDR(a, b) = -<a, b> / (‖a‖‖b‖) (cosinus),
    z = x - y (bruit, ici l'accompagnement), ẑ = x - ŷ, α = ‖y‖² / (‖y‖² + ‖z‖²). Bornée dans [-1, 1].

    Calculée par exemple ET par canal (comme le SI-SDR de la validation) puis moyennée. Le terme de bruit ẑ = x - ŷ rend la loss
    sensible à l'échelle de ŷ, contrairement au SI-SDR (le papier : « solve the scale insensitivity problem »)."""
    y, y_hat = target.reshape(-1, target.shape[-1]), est.reshape(-1, est.shape[-1])
    z, z_hat = mixture.reshape(-1, mixture.shape[-1]) - y, mixture.reshape(-1, mixture.shape[-1]) - y_hat

    def neg_cos(a, b):
        return -(a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + EPS)

    ey, ez = (y ** 2).sum(-1), (z ** 2).sum(-1)
    alpha = ey / (ey + ez + EPS)
    return (alpha * neg_cos(y, y_hat) + (1 - alpha) * neg_cos(z, z_hat)).mean()


def _stft_mag(x: torch.Tensor, n_fft: int, win_length: int, hop: int) -> torch.Tensor:
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_length, window=torch.hann_window(win_length, device=x.device),
                      return_complex=True)
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + MRSTFT_FLOOR)


def _stft_loss_one(est: torch.Tensor, target: torch.Tensor, n_fft: int, win_length: int, hop: int) -> torch.Tensor:
    with torch.no_grad():  # la cible n'a pas de gradient : ne pas garder ses intermédiaires
        mag_tgt = _stft_mag(target, n_fft, win_length, hop)
    mag_est = _stft_mag(est, n_fft, win_length, hop)
    l_sc = (torch.linalg.norm(mag_tgt - mag_est, dim=(-2, -1)) / torch.linalg.norm(mag_tgt, dim=(-2, -1))).mean()  # par signal
    l_mag = torch.nn.functional.l1_loss(torch.log(mag_est), torch.log(mag_tgt))
    return l_sc + l_mag


def multires_stft_loss(est: torch.Tensor, target: torch.Tensor, resolutions=MRSTFT_RESOLUTIONS) -> torch.Tensor:
    """Moyenne sur les résolutions de L_sc + L_mag (Yamamoto et al., eq. 3-6), poids 1 : le papier somme les deux termes sans
    pondération.

        L_sc  = ‖|S(y)| - |S(ŷ)|‖_F / ‖|S(y)|‖_F          (convergence spectrale : relatif, donc insensible au niveau)
        L_mag = (1/N) ‖log|S(y)| - log|S(ŷ)|‖_1           (magnitude log : égalise les bins forts et faibles)

    Ne regarde que des magnitudes : la phase n'est contrainte qu'indirectement, via la cohérence de l'iSTFT.

    Mémoire : la résolution au pas de 92 échantillons produit ~2800 trames ; sans précaution les intermédiaires pèsent ~2 Gio sur
    les 4 Gio disponibles. Chaque résolution est donc recalculée à la rétropropagation (checkpoint) au lieu d'être stockée."""
    est, target = est.reshape(-1, est.shape[-1]), target.reshape(-1, target.shape[-1])
    total = 0.0
    for n_fft, win_length, hop in resolutions:
        if est.requires_grad:
            total = total + checkpoint(_stft_loss_one, est, target, n_fft, win_length, hop, use_reentrant=False)
        else:
            total = total + _stft_loss_one(est, target, n_fft, win_length, hop)
    return total / len(resolutions)
