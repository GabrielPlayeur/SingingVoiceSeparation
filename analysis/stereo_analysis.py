"""Why stereo can help: the vocals are (almost) in the centre, the accompaniment is wide.

Mid/Side decomposition: M = (L + R) / 2 (this is exactly the pipeline's mono), S = (L - R) / 2.
A source panned to the centre has S ≈ 0; a wide or lateral source has a non-negligible S.

Usage: python -m analysis.stereo_analysis [--example "Actions - One Minute Smile"] [--window-seconds 20]
Outputs: figures/stereo_spectrograms.png, figures/stereo_statistics.png + text summary.
"""
import argparse

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from ._common import *

SR = 44100
N_FFT, HOP = 2048, 512


def read_window(d, stem, seconds, center=0.5):
    n = sf.info(f"{d}/{stem}.wav").frames
    length = int(seconds * SR)
    start = max(int(center * n - length / 2), 0)
    x, _ = sf.read(f"{d}/{stem}.wav", start=start, frames=length, dtype="float32", always_2d=True)
    return x.T  # (2, T)


def ms(x):
    return (x[0] + x[1]) / 2, (x[0] - x[1]) / 2


def power_spectrum(x):
    return (np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2).mean(axis=1)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="Actions - One Minute Smile")
    ap.add_argument("--window-seconds", type=float, default=20.0)
    args = ap.parse_args()
    setup_style()

    dirs = track_dirs()
    train_dirs = dirs["train"] + dirs["val"]  # the test set is not analysed: no decision to draw from it

    # ---------- 1) spectrograms of one track: L, R, Mid, Side for vocals and accompaniment
    ex = next(d for d in train_dirs if name_of(d) == args.example)
    seg = 8.0
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for r, stem in enumerate(("vocals", "accompaniment")):
        x = read_window(ex, stem, seg, center=0.5)
        L, R = x
        M, S = ms(x)
        sigs = (("L", L), ("R", R), ("Mid (= mono)", M), ("Side (L − R)/2", S))
        mags = [np.abs(librosa.stft(sig, n_fft=N_FFT, hop_length=HOP)) for _, sig in sigs]
        ref = max(m.max() for m in mags)  # same reference for the 4 panels of a row: comparable levels
        for c, ((lab, _), mag) in enumerate(zip(sigs, mags)):
            ax = axes[r, c]
            im = ax.imshow(20 * np.log10(np.maximum(mag, 1e-10) / ref), origin="lower", aspect="auto", cmap="magma",
                           vmin=-70, vmax=0, extent=[0, seg, 0, SR / 2 / 1000])
            ax.grid(False)
            ax.set_title(f"{'Vocals' if r == 0 else 'Accompaniment'}: {lab}" , fontsize=10)
            if c == 0:
                ax.set_ylabel("frequency (kHz)")
            if r == 1:
                ax.set_xlabel("time (s)")
    ax.set_ylim(0, 8)
    fig.colorbar(im, ax=axes, label="dB (re. row max)", shrink=0.8)
    fig.suptitle(f"{args.example}: Mid / Side spectrograms", fontweight="bold")
    save(fig, "stereo_spectrograms.png")

    # ---------- 2) statistics over all training + validation tracks
    band_edges = np.array([0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 22050])
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    sm_ratio = {"vocals": [], "accompaniment": []}  # S/M per band (dB)
    corr = {"vocals": [], "accompaniment": []}
    side_frac = {"vocals": [], "accompaniment": []}
    kept = 0
    for d in train_dirs:
        x = {s: read_window(d, s, args.window_seconds) for s in sm_ratio}
        if np.abs(x["vocals"]).mean() < 1e-3:  # same threshold as silence_threshold in the configs
            continue
        kept += 1
        for s, sig in x.items():
            M, S = ms(sig)
            pm, ps = power_spectrum(M), power_spectrum(S)
            ratios = []
            for lo, hi in zip(band_edges[:-1], band_edges[1:]):
                k = (freqs >= lo) & (freqs < hi)
                ratios.append(db(ps[k].sum() / pm[k].sum()))
            sm_ratio[s].append(ratios)
            corr[s].append(np.corrcoef(sig[0], sig[1])[0, 1])
            side_frac[s].append((S ** 2).sum() / ((M ** 2).sum() + (S ** 2).sum()))
    print(f"{kept}/{len(train_dirs)} tracks analysed (central window of {args.window_seconds:.0f} s, non-silent vocals)")
    for s in sm_ratio:
        c, f, r = np.array(corr[s]), np.array(side_frac[s]), np.array(sm_ratio[s])
        print(f"{s:14s} L/R correlation: median {np.median(c):.3f} (10th–90th perc. {np.percentile(c, 10):.3f}–{np.percentile(c, 90):.3f}); "
              f"Side energy: median {np.median(f):.1%}; overall median S/M {np.median(db(f / (1 - f))):.1f} dB")
        print("   median S/M per band (dB):", {f"{lo}-{hi}": round(float(np.median(r[:, i])), 1) for i, (lo, hi) in enumerate(zip(band_edges[:-1], band_edges[1:]))})
    cv, ca = np.array(corr["vocals"]), np.array(corr["accompaniment"])
    print(f"Correlation vocals > accompaniment on {(cv > ca).mean():.0%} of the tracks")
    print(f"Nearly mono vocals (correlation > 0.95): {(cv > 0.95).mean():.0%} of the tracks; accompaniment: {(ca > 0.95).mean():.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    ax = axes[0]
    centers = np.arange(len(band_edges) - 1)
    for s, color, lab in (("vocals", BLUE, "vocals"), ("accompaniment", ORANGE, "accompaniment")):
        r = np.array(sm_ratio[s])
        lo, med, hi = np.percentile(r, [25, 50, 75], axis=0)
        ax.fill_between(centers, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(centers, med, color=color, linewidth=2.2, marker="o", markersize=5, label=lab)
    ax.set_xticks(centers, [f"{int(lo)}–{int(hi)}" for lo, hi in zip(band_edges[:-1], band_edges[1:])], rotation=30)
    ax.set_xlabel("frequency band (Hz)")
    ax.set_ylabel("Side energy / Mid energy (dB)")
    ax.set_title("Side / Mid by band (median, 25–75 %)")
    ax.legend()

    ax = axes[1]
    bins = np.linspace(-0.2, 1.0, 25)
    ax.hist(cv, bins=bins, color=BLUE, alpha=0.75, label="vocals", edgecolor="white")
    ax.hist(ca, bins=bins, color=ORANGE, alpha=0.75, label="accompaniment", edgecolor="white")
    ax.set_xlabel("correlation")
    ax.set_ylabel("tracks")
    ax.set_title("Left–right correlation (1 = mono)")
    ax.legend(loc="upper left")
    fig.suptitle(f"Stereo cues ({kept} train + val tracks)", fontweight="bold")
    save(fig, "stereo_statistics.png")


if __name__ == "__main__":
    main()
