"""Gap between the official mixture (mixture.wav, encoded separately in AAC) and the sum vocals + accompaniment.

The pipeline trains and evaluates on `vocals + accompaniment` (see MODEL_WALKTHROUGH § 1.2). This script measures:
  1. the size of the gap per track (SNR of the residual `mixture - (vocals + accompaniment)`);
  2. its distribution over frequency;
  3. its effect on an oracle: mask computed on the rebuilt mixture, applied to the true mixture;
     SI-SDR of the estimated vocals, compared with the oracle on the rebuilt mixture.

Usage: python -m analysis.mixture_gap [--crops-per-track 3] [--skip-oracle]
Outputs: figures/mixture_gap.png, figures/mixture_gap.csv
"""
import argparse
import csv
import os

import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt

from ._common import *
from src.models.oracle import Oracle
from src.utils.utils import si_sdr

N_FFT = 2048
BLOCK = 44100 * 10


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))


def scan_track(d):
    """Energies and mean spectra (power per bin) of the mixture and of the residual, read in blocks of 10 s."""
    e_mix = e_voc = e_res = 0.0
    p_mix = np.zeros(N_FFT // 2 + 1)
    p_res = np.zeros(N_FFT // 2 + 1)
    win = np.hanning(N_FFT)
    n_frames = 0
    n = min(sf.info(f"{d}/{f}.wav").frames for f in ("mixture", "vocals", "accompaniment"))
    readers = [sf.blocks(f"{d}/{f}.wav", blocksize=BLOCK, dtype="float32", always_2d=True)
               for f in ("mixture", "vocals", "accompaniment")]
    seen = 0
    for mix, voc, acc in zip(*readers):
        m = min(len(mix), len(voc), len(acc), n - seen)
        mix, voc, acc = mix[:m], voc[:m], acc[:m]
        seen += m
        res = mix - (voc + acc)
        e_mix += float((mix.astype(np.float64) ** 2).sum())
        e_voc += float((voc.astype(np.float64) ** 2).sum())
        e_res += float((res.astype(np.float64) ** 2).sum())
        k = m // N_FFT
        if k:
            for x, p in ((mix, p_mix), (res, p_res)):
                frames = x[:k * N_FFT].mean(axis=1).reshape(k, N_FFT) * win
                p += (np.abs(np.fft.rfft(frames, axis=1)) ** 2).sum(axis=0)
            n_frames += k
    return dict(e_mix=e_mix, e_voc=e_voc, e_res=e_res, p_mix=p_mix / n_frames, p_res=p_res / n_frames, n=seen)


def oracle_sensitivity(d, oracle, crop_len, positions):
    """Oracle SI-SDR (dB): mask derived from the rebuilt mixture, applied (a) to the rebuilt mixture, (b) to the true mixture."""
    n = sf.info(f"{d}/mixture.wav").frames
    out = []
    for pos in positions:
        start = int(pos * (n - crop_len))
        rd = lambda f: sf.read(f"{d}/{f}.wav", start=start, frames=crop_len, dtype="float32", always_2d=True)[0].mean(axis=1)
        voc, acc, real = rd("vocals"), rd("accompaniment"), rd("mixture")
        if np.abs(voc).mean() < 1e-3:  # same anti-silence filter as the dataset (silence_threshold)
            continue
        recon = voc + acc
        v_a = oracle.evaluate(recon, voc)
        # (b) same mask, applied to the true mixture: the mask is rebuilt by hand to be reused
        v_b = oracle_apply_mask(oracle, recon, voc, real)
        t = torch.from_numpy(voc)[None]
        out.append((float(si_sdr(torch.from_numpy(v_a)[None], t)), float(si_sdr(torch.from_numpy(v_b)[None], t))))
    return out


def oracle_apply_mask(oracle, recon, voc, target_mix):
    import librosa
    kw = dict(n_fft=oracle.n_fft, hop_length=oracle.hop_length, pad_mode="reflect")
    mask = np.clip(np.abs(librosa.stft(voc, **kw)) / (np.abs(librosa.stft(recon, **kw)) + 1e-8), 0, 1)
    spec = librosa.stft(target_mix, **kw)
    return librosa.istft(mask * spec, hop_length=oracle.hop_length, length=len(target_mix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-per-track", type=int, default=3)
    ap.add_argument("--crop-seconds", type=float, default=6.0)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--max-tracks", type=int, default=None, help="to test quickly")
    args = ap.parse_args()
    setup_style()

    dirs = track_dirs()
    all_dirs = [(s, d) for s in ("train", "val", "test") for d in dirs[s]][:args.max_tracks]
    oracle = Oracle(n_fft=2048, hop_length=512)
    crop_len = int(44100 * args.crop_seconds)
    positions = np.linspace(0.2, 0.8, args.crops_per_track)

    rows, spec_ratio = [], []
    for i, (subset, d) in enumerate(all_dirs):
        s = scan_track(d)
        row = dict(track=name_of(d), subset=subset,
                   snr_mix_db=db(s["e_mix"] / s["e_res"]), snr_voc_db=db(s["e_voc"] / s["e_res"]),
                   duration_s=s["n"] / 44100)
        spec_ratio.append(db(s["p_res"] / s["p_mix"]))
        if not args.skip_oracle and subset == "test":
            res = oracle_sensitivity(d, oracle, crop_len, positions)
            row["oracle_recon"] = float(np.mean([r[0] for r in res])) if res else np.nan
            row["oracle_real"] = float(np.mean([r[1] for r in res])) if res else np.nan
        rows.append(row)
        print(f"[{i + 1}/{len(all_dirs)}] {row['track'][:45]:45s} SNR(mixture/residual) = {row['snr_mix_db']:5.1f} dB", flush=True)

    os.makedirs(FIG_DIR, exist_ok=True)
    cols = ["track", "subset", "duration_s", "snr_mix_db", "snr_voc_db", "oracle_recon", "oracle_real"]
    with open(os.path.join(FIG_DIR, "mixture_gap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    snr = np.array([r["snr_mix_db"] for r in rows])
    snr_voc = np.array([r["snr_voc_db"] for r in rows])
    print(f"\nSNR mixture/residual: median {np.median(snr):.1f} dB, min {snr.min():.1f}, max {snr.max():.1f}, "
          f"10e percentile {np.percentile(snr, 10):.1f}")
    print(f"SNR vocals/residual : median {np.median(snr_voc):.1f} dB, min {snr_voc.min():.1f}")
    for s in ("train", "val", "test"):
        v = np.array([r["snr_mix_db"] for r in rows if r["subset"] == s])
        if len(v):
            print(f"  {s:5s} median {np.median(v):5.1f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ax = axes[0]
    bins = np.linspace(snr.min() - 1, snr.max() + 1, 24)
    ax.hist(snr, bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(np.median(snr), color=INK, linewidth=1.5)
    ax.text(np.median(snr), ax.get_ylim()[1] * 0.95, f" median {np.median(snr):.1f} dB", va="top", color=INK_2)
    ax.set_xlabel("residual SNR: 10·log10(‖mixture‖² / ‖mixture − (vocals + accomp.)‖²)  [dB]")
    ax.set_ylabel("tracks")
    ax.set_title(f"Gap per track ({len(rows)} tracks)")

    ax = axes[1]
    sr_ratio = np.array(spec_ratio)
    freqs = np.fft.rfftfreq(N_FFT, 1 / 44100) / 1000
    lo, med, hi = np.percentile(sr_ratio, [10, 50, 90], axis=0)
    ax.fill_between(freqs, lo, hi, color=BLUE, alpha=0.2, linewidth=0)
    ax.plot(freqs, med, color=BLUE, linewidth=2)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("residual / mixture (dB)")
    ax.set_title("Frequency distribution (median, 10–90 % band)")

    ax = axes[2]
    both = [(r["oracle_recon"], r["oracle_real"]) for r in rows if "oracle_recon" in r and not np.isnan(r["oracle_recon"])]
    if both:
        a, b = np.array(both).T
        ax.scatter(a, b, color=ORANGE, s=28, edgecolor="white", linewidth=0.8, zorder=3)
        lim = [min(a.min(), b.min()) - 1, max(a.max(), b.max()) + 1]
        ax.plot(lim, lim, color=INK_2, linewidth=1, linestyle="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("oracle on rebuilt mixture (SI-SDR, dB)")
        ax.set_ylabel("same mask on the true mixture (dB)")
        ax.set_title(f"Effect on the oracle ({len(both)} test tracks)")
        print(f"Oracle SI-SDR: rebuilt {np.median(a):.2f} dB, true mixture {np.median(b):.2f} dB, "
              f"median gap {np.median(a - b):.2f} dB")
    else:
        ax.axis("off")
    fig.suptitle("Compressed MUSDB18: the official mixture is not the sum of the stems", fontweight="bold")
    save(fig, "mixture_gap.png")


if __name__ == "__main__":
    main()
