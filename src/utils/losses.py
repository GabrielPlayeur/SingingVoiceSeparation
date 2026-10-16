"""Losses computed on the estimated WAVEFORM (ROADMAP, step 2b, walkthrough limit e).

The waveform is `istft(mask * |X|, mixture phase)` (`VoiceSeparationModule._to_wave`, differentiable): the loss therefore sees
the actual reconstruction, mixture phase included, i.e. what the evaluation measures. All of them take (B, C, T) tensors and
return a scalar to MINIMISE.

    sisdr   : -SI-SDR (dB)                                        Le Roux et al. 2019, eq. 4-5
    wsdr    : -[α·cos(y, ŷ) + (1-α)·cos(z, ẑ)], z = x - y         Choi et al. 2019 (ICLR), eq. 7 and 9
    mrstft  : mean over M resolutions of (L_sc + L_mag)         Yamamoto et al. 2020, eq. 3-6 and table 1

Nothing here comes from a library: the formulas are those of the papers, read in full (see README).
"""
import torch
from torch.utils.checkpoint import checkpoint

from .utils import si_sdr

EPS = 1e-8  # denominators of wsdr: "we add small number ε to denominators" (Choi et al., appendix C); the value is chosen here

# Table 1 of Yamamoto et al. (24 kHz): (FFT, window, hop) = (1024, 600, 120), (2048, 1200, 240), (512, 240, 50), i.e.
# windows of 25 / 50 / 10 ms and hops of 5 / 10 / ~2 ms. Converted to 44.1 kHz KEEPING THE DURATIONS (⚠ c: adaptation made
# here, the paper only covers 24 kHz), FFT = power of 2 that contains the window.
MRSTFT_RESOLUTIONS = (
    (2048, 1102, 220),  # window 25 ms, hop 5 ms
    (4096, 2205, 441),  # window 50 ms, hop 10 ms
    (1024, 441, 92),    # window 10 ms, hop ~2 ms
)
MRSTFT_FLOOR = 1e-7  # floor of |STFT|²: d|z|/dz is undefined at 0; implementation choice, not in the paper

LOSS_TYPES = ("amp_l1", "sisdr", "wsdr", "mrstft")


def sisdr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean -SI-SDR over (examples, channels). Same function as the validation metric (`utils.si_sdr`), hence the same formula
    as `val/si_sdr_*`: exactly what is monitored is optimised. Invariant to the scale of `est` (Le Roux et al., § 2.3)."""
    return -si_sdr(est, target).mean()


def wsdr_loss(mixture: torch.Tensor, target: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
    """Weighted-SDR of Choi et al.: α·loss_SDR(y, ŷ) + (1-α)·loss_SDR(z, ẑ), loss_SDR(a, b) = -<a, b> / (‖a‖‖b‖) (cosine),
    z = x - y (noise, here the accompaniment), ẑ = x - ŷ, α = ‖y‖² / (‖y‖² + ‖z‖²). Bounded in [-1, 1].

    Computed per example AND per channel (like the validation SI-SDR) then averaged. The noise term ẑ = x - ŷ makes the loss
    sensitive to the scale of ŷ, unlike SI-SDR (the paper: "solve the scale insensitivity problem")."""
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
    with torch.no_grad():  # the target has no gradient: do not keep its intermediates
        mag_tgt = _stft_mag(target, n_fft, win_length, hop)
    mag_est = _stft_mag(est, n_fft, win_length, hop)
    l_sc = (torch.linalg.norm(mag_tgt - mag_est, dim=(-2, -1)) / torch.linalg.norm(mag_tgt, dim=(-2, -1))).mean()  # per signal
    l_mag = torch.nn.functional.l1_loss(torch.log(mag_est), torch.log(mag_tgt))
    return l_sc + l_mag


def multires_stft_loss(est: torch.Tensor, target: torch.Tensor, resolutions=MRSTFT_RESOLUTIONS) -> torch.Tensor:
    """Mean over the resolutions of L_sc + L_mag (Yamamoto et al., eq. 3-6), weight 1: the paper sums the two terms without
    weighting.

        L_sc  = ‖|S(y)| - |S(ŷ)|‖_F / ‖|S(y)|‖_F          (spectral convergence: relative, hence insensitive to the level)
        L_mag = (1/N) ‖log|S(y)| - log|S(ŷ)|‖_1           (log magnitude: balances strong and weak bins)

    Only looks at magnitudes: the phase is constrained only indirectly, through the consistency of the iSTFT.

    Memory: the resolution with a hop of 92 samples produces ~2800 frames; without care the intermediates weigh ~2 GiB out of
    the 4 GiB available. Each resolution is therefore recomputed during backpropagation (checkpoint) instead of being stored."""
    est, target = est.reshape(-1, est.shape[-1]), target.reshape(-1, target.shape[-1])
    total = 0.0
    for n_fft, win_length, hop in resolutions:
        if est.requires_grad:
            total = total + checkpoint(_stft_loss_one, est, target, n_fft, win_length, hop, use_reentrant=False)
        else:
            total = total + _stft_loss_one(est, target, n_fft, win_length, hop)
    return total / len(resolutions)
