"""Pourquoi la stéréo peut aider : la voix est (presque) au centre, l'accompagnement est large.

Décomposition Mid/Side : M = (L + R) / 2 (c'est exactement le mono du pipeline), S = (L - R) / 2.
Une source panoramiquée au centre a S ≈ 0 ; une source large ou latérale a S non négligeable.

Usage : python -m analysis.stereo_analysis [--example "Actions - One Minute Smile"] [--window-seconds 20]
Sorties : figures/stereo_spectrograms.png, figures/stereo_statistics.png + résumé texte.
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
    train_dirs = dirs["train"] + dirs["val"]  # on n'analyse pas le test : pas de décision à en tirer

    # ---------- 1) spectrogrammes d'un morceau : L, R, Mid, Side pour voix et accompagnement
    ex = next(d for d in train_dirs if name_of(d) == args.example)
    seg = 8.0
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for r, stem in enumerate(("vocals", "accompaniment")):
        x = read_window(ex, stem, seg, center=0.5)
        L, R = x
        M, S = ms(x)
        sigs = (("L", L), ("R", R), ("Mid (= mono)", M), ("Side (L − R)/2", S))
        mags = [np.abs(librosa.stft(sig, n_fft=N_FFT, hop_length=HOP)) for _, sig in sigs]
        ref = max(m.max() for m in mags)  # même référence pour les 4 panneaux d'une ligne : niveaux comparables
        for c, ((lab, _), mag) in enumerate(zip(sigs, mags)):
            ax = axes[r, c]
            im = ax.imshow(20 * np.log10(np.maximum(mag, 1e-10) / ref), origin="lower", aspect="auto", cmap="magma",
                           vmin=-70, vmax=0, extent=[0, seg, 0, SR / 2 / 1000])
            ax.grid(False)
            ax.set_title(f"{'Voix' if r == 0 else 'Accompagnement'} : {lab}" , fontsize=10)
            if c == 0:
                ax.set_ylabel("fréquence (kHz)")
            if r == 1:
                ax.set_xlabel("temps (s)")
    ax.set_ylim(0, 8)
    fig.colorbar(im, ax=axes, label="amplitude STFT (dB, réf. = max de la ligne)", shrink=0.8)
    fig.suptitle(f"{args.example} : la voix tient surtout dans Mid, l'accompagnement a une énergie Side importante", fontweight="bold")
    save(fig, "stereo_spectrograms.png")

    # ---------- 2) statistiques sur tous les morceaux d'entraînement + validation
    band_edges = np.array([0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 22050])
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    sm_ratio = {"vocals": [], "accompaniment": []}  # S/M par bande (dB)
    corr = {"vocals": [], "accompaniment": []}
    side_frac = {"vocals": [], "accompaniment": []}
    kept = 0
    for d in train_dirs:
        x = {s: read_window(d, s, args.window_seconds) for s in sm_ratio}
        if np.abs(x["vocals"]).mean() < 1e-3:  # même seuil que silence_threshold des configs
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
    print(f"{kept}/{len(train_dirs)} morceaux analysés (fenêtre centrale de {args.window_seconds:.0f} s, voix non silencieuse)")
    for s in sm_ratio:
        c, f, r = np.array(corr[s]), np.array(side_frac[s]), np.array(sm_ratio[s])
        print(f"{s:14s} corrélation L/R : médiane {np.median(c):.3f} (10e–90e perc. {np.percentile(c, 10):.3f}–{np.percentile(c, 90):.3f}) ; "
              f"énergie dans Side : médiane {np.median(f):.1%} ; S/M global médian {np.median(db(f / (1 - f))):.1f} dB")
        print("   S/M médian par bande (dB) :", {f"{lo}-{hi}": round(float(np.median(r[:, i])), 1) for i, (lo, hi) in enumerate(zip(band_edges[:-1], band_edges[1:]))})
    cv, ca = np.array(corr["vocals"]), np.array(corr["accompaniment"])
    print(f"Corrélation voix > accompagnement sur {(cv > ca).mean():.0%} des morceaux")
    print(f"Voix quasi mono (corrélation > 0.95) : {(cv > 0.95).mean():.0%} des morceaux ; accompagnement : {(ca > 0.95).mean():.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    ax = axes[0]
    centers = np.arange(len(band_edges) - 1)
    for s, color, lab in (("vocals", BLUE, "voix"), ("accompaniment", ORANGE, "accompagnement")):
        r = np.array(sm_ratio[s])
        lo, med, hi = np.percentile(r, [25, 50, 75], axis=0)
        ax.fill_between(centers, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(centers, med, color=color, linewidth=2.2, marker="o", markersize=5, label=lab)
    ax.set_xticks(centers, [f"{int(lo)}–{int(hi)}" for lo, hi in zip(band_edges[:-1], band_edges[1:])], rotation=30)
    ax.set_xlabel("bande de fréquence (Hz)")
    ax.set_ylabel("énergie Side / énergie Mid (dB)")
    ax.set_title("Largeur stéréo par bande (médiane, bande 25–75 %)")
    ax.legend()

    ax = axes[1]
    bins = np.linspace(-0.2, 1.0, 25)
    ax.hist(cv, bins=bins, color=BLUE, alpha=0.75, label="voix", edgecolor="white")
    ax.hist(ca, bins=bins, color=ORANGE, alpha=0.75, label="accompagnement", edgecolor="white")
    ax.set_xlabel("corrélation entre les canaux L et R")
    ax.set_ylabel("morceaux")
    ax.set_title("Une corrélation de 1 = source strictement mono")
    ax.legend(loc="upper left")
    fig.suptitle(f"Indices spatiaux dans MUSDB18 ({kept} morceaux d'entraînement + validation)", fontweight="bold")
    save(fig, "stereo_statistics.png")


if __name__ == "__main__":
    main()
