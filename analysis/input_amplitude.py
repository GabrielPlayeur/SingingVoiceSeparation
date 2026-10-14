"""Amplitudes d'entrée du U-Net et sensibilité au volume (ROADMAP 2a, limite B10 du walkthrough).

Hypothèse testée : monter ou baisser le volume d'un morceau ne devrait pas changer le masque, donc la qualité de la séparation.
Or le réseau reçoit |X| brut (aucune normalisation, médiane 0,053, maximum de l'ordre de 200) et sa BatchNorm utilise en
inférence des statistiques glissantes figées : le masque dépend du niveau absolu.

Deux figures :
  figures/input_amplitude_levels.png  : ce que reçoit le réseau (dynamique des bins, niveaux par morceau, niveau par fréquence)
  figures/input_gain_sensitivity.png  : (a) décalage de l'entrée du réseau sous un gain, pour chaque transformation d'entrée ;
                                        (b, c) pour chaque checkpoint : SI-SDR de validation et variation du masque selon le gain

Usage :
    python -m analysis.input_amplitude [--ckpt mono=CKPT --ckpt stereo=CKPT ...] [--crops-per-track 5]
Sans --ckpt, les deux baselines (mono ygdtir6a, stéréo fgvzjxua) sont utilisées si elles existent. Pour ajouter un modèle
entraîné avec une compression d'entrée : --ckpt stereo_log1p=outputs/VoiceSeparation/<run>/checkpoints/<ckpt>.
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from ._common import *
from src.data.data import MUSDB18DataModule
from src.module import VoiceSeparationModule
from src.utils.utils import si_sdr
from src.utils.input_transform import INPUT_TRANSFORMS, transform_amplitude

GAINS_DB = [-18, -12, -6, -3, 0, 3, 6, 12, 18]
DEFAULT_CKPTS = {"mono": "ygdtir6a", "stereo": "fgvzjxua"}
LEVEL_BINS = np.linspace(-120, 60, 181)  # dB, pour l'histogramme des amplitudes


def stft_amp(x, n_fft=2048, hop=512):
    """x (B, C, T) -> |STFT| (B, C, F, T'), mêmes réglages que le module (fenêtre de Hann, center=True)."""
    B, C, T = x.shape
    spec = torch.stft(x.reshape(B * C, T), n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                      return_complex=True, center=True)
    return spec.abs().reshape(B, C, *spec.shape[-2:])


def load_val_crops(crops_per_track):
    """Extraits de validation (mêmes morceaux et même filtre anti-silence que l'entraînement), en stéréo : (N, 2, T) x 2."""
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(ROOT, "src/configs/data/musdb.yaml")), resolve=True)
    cfg.update(channels=2, val_crops_per_track=crops_per_track)
    cfg["dataset_path"] = os.path.join(ROOT, cfg["dataset_path"])
    cfg["split_path"] = os.path.join(ROOT, cfg["split_path"])
    dm = MUSDB18DataModule(**cfg)
    dm.setup("fit")
    crops = dm.val_dataset.crops
    mixture = torch.from_numpy(np.stack([c["mixture"] for c in crops])).float()
    vocals = torch.from_numpy(np.stack([c["vocals"] for c in crops])).float()
    return mixture, vocals


def track_levels(seconds=6.0):
    """Niveau moyen de |X| (dB) d'un extrait central de chaque morceau d'entraînement + validation : variabilité entre morceaux."""
    dirs = track_dirs()
    levels, names = [], []
    for d in dirs["train"] + dirs["val"]:
        n = sf.info(f"{d}/mixture.wav").frames
        length = int(seconds * 44100)
        start = max(n // 2 - length // 2, 0)
        v = sf.read(f"{d}/vocals.wav", start=start, frames=length, dtype="float32", always_2d=True)[0]
        a = sf.read(f"{d}/accompaniment.wav", start=start, frames=length, dtype="float32", always_2d=True)[0]
        mono = torch.from_numpy((v + a).mean(axis=1))[None, None]
        levels.append(20 * np.log10(stft_amp(mono).mean().item() + 1e-12))
        names.append(name_of(d))
    return np.array(levels), names


@torch.no_grad()
def model_sensitivity(module, mixture, vocals, device, batch_size=8):
    """Pour chaque gain : SI-SDR de chaque extrait et variation du masque par rapport au gain 0 dB."""
    module = module.to(device).eval()
    if module.hparams.in_channels == 1:  # même mono que le pipeline : moyenne L/R
        mixture, vocals = mixture.mean(1, keepdim=True), vocals.mean(1, keepdim=True)
    sisdr = {g: [] for g in GAINS_DB}
    dmask = {g: [] for g in GAINS_DB}
    mask_mean = {g: [] for g in GAINS_DB}
    for i in range(0, len(mixture), batch_size):
        mix, voc = mixture[i:i + batch_size].to(device), vocals[i:i + batch_size].to(device)
        ref_mask = None
        for g_db in [0] + [g for g in GAINS_DB if g != 0]:
            batch = module.on_after_batch_transfer({"mixture": mix * 10 ** (g_db / 20), "vocals": voc}, 0)
            mask = module(batch["mix_amp"])
            if g_db == 0:
                ref_mask = mask
            est = module._to_wave(mask * batch["mix_amp"], batch["mix_phase"], voc.shape[-1])
            sisdr[g_db].append(si_sdr(est, voc).cpu())  # invariant à l'échelle de `est` : seul le masque compte
            dmask[g_db].append((mask - ref_mask).abs().mean(dim=(1, 2, 3)).cpu())
            mask_mean[g_db].append(mask.mean(dim=(1, 2, 3)).cpu())
    cat = lambda d: {g: torch.cat(v).numpy() for g, v in d.items()}
    return cat(sisdr), cat(dmask), cat(mask_mean)


def input_shift(amp, kind):
    """Décalage moyen de l'entrée du réseau sous un gain, en écarts-types de l'entrée à 0 dB : c'est ce que voit la BatchNorm."""
    ref = transform_amplitude(amp, kind)
    mean0, std0 = ref.mean().item(), ref.std().item()
    return {g: (transform_amplitude(amp * 10 ** (g / 20), kind).mean().item() - mean0) / std0 for g in GAINS_DB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], metavar="NOM=CHEMIN")
    ap.add_argument("--crops-per-track", type=int, default=5)
    args = ap.parse_args()
    setup_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    if not ckpts:
        for name, run in DEFAULT_CKPTS.items():
            found = glob.glob(os.path.join(ROOT, f"outputs/VoiceSeparation/{run}/checkpoints/*.ckpt"))
            if found:
                ckpts[name] = found[0]

    mixture, vocals = load_val_crops(args.crops_per_track)
    amp = stft_amp(mixture)  # (N, 2, F, T)
    print(f"{len(mixture)} extraits de validation, amplitudes |X| : forme {tuple(amp.shape)}")

    # ---------- statistiques de l'entrée
    pct = {p: float(np.percentile(amp.numpy()[::4], p)) for p in (1, 10, 50, 90, 99, 99.9)}
    print("percentiles de |X| :", {p: round(v, 4) for p, v in pct.items()}, "max", round(float(amp.max()), 1))
    print(f"dynamique médiane -> p99,9 : {20 * np.log10(pct[99.9] / pct[50]):.0f} dB ; médiane -> max : {20 * np.log10(float(amp.max()) / pct[50]):.0f} dB")
    print(f"bins quasi vides (|X| < 1e-3, soit < -60 dB) : {(amp < 1e-3).float().mean():.1%} (au-dessus de ~16 kHz : coupure de l'encodage AAC)")
    print(f"part des bins avec |X| < 1 : {(amp < 1).float().mean():.1%} (log1p y est quasi l'identité)")
    levels, names = track_levels()
    print(f"niveau moyen par morceau : médiane {np.median(levels):.1f} dB, p10-p90 {np.percentile(levels, 10):.1f} à "
          f"{np.percentile(levels, 90):.1f} dB (écart {np.percentile(levels, 90) - np.percentile(levels, 10):.1f} dB), "
          f"min {levels.min():.1f}, max {levels.max():.1f} ({names[int(levels.argmin())]} / {names[int(levels.argmax())]})")

    hist = np.zeros(len(LEVEL_BINS) - 1)
    for a in amp:
        hist += np.histogram(20 * np.log10(np.maximum(a.numpy().ravel(), 1e-6)), bins=LEVEL_BINS)[0]
    hist /= hist.sum()
    freq_level = 20 * np.log10(amp.mean(dim=(0, 1, 3)).numpy() + 1e-12)
    freqs = np.fft.rfftfreq(2048, 1 / 44100) / 1000

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    ax = axes[0]
    centers = (LEVEL_BINS[:-1] + LEVEL_BINS[1:]) / 2
    ax.fill_between(centers, hist, color=BLUE, alpha=0.35, linewidth=0)
    ax.plot(centers, hist, color=BLUE, linewidth=1.5)
    for p, label in ((50, "médiane"), (99.9, "p99,9")):
        x = 20 * np.log10(pct[p])
        ax.axvline(x, color=INK_2, linestyle="--", linewidth=1)
        ax.text(x - 1, ax.get_ylim()[1] * 0.97, f"{label} : |X| = {pct[p]:.3g}", fontsize=8, color=INK_2, va="top", ha="right")
    ax.axvline(0, color=ORANGE, linewidth=1.5)
    ax.text(1, ax.get_ylim()[1] * 0.55, "|X| = 1\nà gauche : log1p(|X|) ≈ |X|", fontsize=8, color=ORANGE)
    ax.set_xlabel("20·log10 |X| (dB)")
    ax.set_ylabel("part des bins")
    ax.set_title(f"Amplitudes reçues par le réseau ({len(mixture)} extraits)")

    ax = axes[1]
    order = np.argsort(levels)
    ax.scatter(np.arange(len(levels)), levels[order], color=BLUE, s=22, edgecolor="white", linewidth=0.6, zorder=3)
    lo, hi = np.percentile(levels, [10, 90])
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.18)
    ax.text(1, hi + 0.4, f"p10–p90 : {hi - lo:.1f} dB d'écart entre morceaux", fontsize=8.5, color=INK_2)
    ax.set_xlabel("morceaux (triés)")
    ax.set_ylabel("niveau moyen de |X| (dB)")
    ax.set_title(f"Volume par morceau ({len(levels)} extraits centraux de 6 s)")

    ax = axes[2]
    ax.plot(freqs, freq_level, color=BLUE, linewidth=1.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 22.05)
    ax.set_xlabel("fréquence (kHz)")
    ax.set_ylabel("niveau moyen de |X| (dB)")
    ax.set_title("Niveau par fréquence (la L1 est dominée par les graves)")
    fig.suptitle("Ce que reçoit le U-Net : amplitudes brutes, sans normalisation", fontweight="bold")
    save(fig, "input_amplitude_levels.png")

    # ---------- sensibilité au gain
    shifts = {k: input_shift(amp[:20], k) for k in INPUT_TRANSFORMS}
    print("\ndécalage moyen de l'entrée du réseau à +6 dB, en écarts-types de l'entrée à 0 dB :",
          {k: round(v[6], 2) for k, v in shifts.items()})
    results = {}
    for name, path in ckpts.items():
        module = VoiceSeparationModule.load_from_checkpoint(path, map_location="cpu")
        sis, dm, mm = model_sensitivity(module, mixture, vocals, device)
        results[name] = {"input_transform": module.hparams.input_transform, "ckpt": os.path.relpath(path, ROOT),
                         "si_sdr_median": {g: float(np.median(sis[g])) for g in GAINS_DB},
                         "si_sdr_gain_vs_0": {g: float(np.median(sis[g] - sis[0])) for g in GAINS_DB},
                         "delta_mask": {g: float(dm[g].mean()) for g in GAINS_DB},
                         "mask_mean": {g: float(mm[g].mean()) for g in GAINS_DB}}
        r = results[name]
        print(f"\n{name} ({module.hparams.input_transform}, {r['ckpt']})")
        print("  gain (dB)       " + " ".join(f"{g:6d}" for g in GAINS_DB))
        print("  SI-SDR médian   " + " ".join(f"{r['si_sdr_median'][g]:6.2f}" for g in GAINS_DB))
        print("  Δ SI-SDR vs 0dB " + " ".join(f"{r['si_sdr_gain_vs_0'][g]:6.2f}" for g in GAINS_DB))
        print("  |Δ masque|      " + " ".join(f"{r['delta_mask'][g]:6.3f}" for g in GAINS_DB))
        print("  masque moyen    " + " ".join(f"{r['mask_mean'][g]:6.3f}" for g in GAINS_DB))
    with open(os.path.join(FIG_DIR, "input_gain_sensitivity.json"), "w") as f:
        json.dump({"gains_db": GAINS_DB, "input_shift": {k: {str(g): v for g, v in d.items()} for k, d in shifts.items()},
                   "models": results}, f, indent=1)

    colors = {"mono": BLUE, "stereo": ORANGE}
    palette = [BLUE, ORANGE, AQUA, INK_2]
    # panneau (a) : une couleur par transformation, distincte du bleu/orange (= modèles mono/stéréo) des panneaux (b) et (c)
    transform_style = {"none": (INK_2, "o"), "log1p": ("#eda100", "s"), "sqrt": ("#6250d6", "^"), "norm": (AQUA, "D")}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    ax = axes[0]
    for k in INPUT_TRANSFORMS:
        c, marker = transform_style[k]
        ax.plot(GAINS_DB, [shifts[k][g] for g in GAINS_DB], color=c, linewidth=2, marker=marker, markersize=5, label=k)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("gain appliqué au morceau (dB)")
    ax.set_ylabel("décalage de l'entrée du réseau\n(en écarts-types de l'entrée à 0 dB)")
    ax.set_title("(a) Ce que voit la BatchNorm selon la transformation")
    ax.legend(title="input_transform")

    for ax, key, ylabel, title in ((axes[1], "si_sdr_gain_vs_0", "SI-SDR médian − SI-SDR à 0 dB (dB)", "(b) Qualité de séparation selon le gain"),
                                   (axes[2], "delta_mask", "écart moyen |masque(g) − masque(0 dB)|", "(c) Le masque change-t-il avec le volume ?")):
        for i, (name, r) in enumerate(results.items()):
            ax.plot(GAINS_DB, [r[key][g] for g in GAINS_DB], color=colors.get(name, palette[i % 4]), linewidth=2, marker="o",
                    markersize=4, label=f"{name} ({r['input_transform']})")
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_xlabel("gain appliqué au morceau (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Sensibilité au volume : un modèle invariant resterait sur la ligne 0", fontweight="bold")
    save(fig, "input_gain_sensitivity.png")


if __name__ == "__main__":
    main()
