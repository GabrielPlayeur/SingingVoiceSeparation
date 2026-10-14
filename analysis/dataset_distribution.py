"""Répartition de MUSDB18 (genre, origine, artiste, durée) par sous-ensemble train / val / test.

Usage : python -m analysis.dataset_distribution
Sorties : figures/dataset_distribution.png, figures/dataset_artists.png + tableau texte.
"""
from collections import Counter

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from ._common import *


SHORT_ORIGIN = {"C't Remix Comp.": "Heise remix", "Native Instruments": "Native Instr."}


def main():
    setup_style()
    meta = read_metadata()
    dirs = track_dirs()
    subsets = ["train", "val", "test"]
    colors = {"train": BLUE, "val": AQUA, "test": ORANGE}

    names = {s: [name_of(d) for d in dirs[s]] for s in subsets}
    missing = [n for s in subsets for n in names[s] if meta[n]["genre"] == "Inconnu"]
    if missing:
        print("Sans métadonnées dans le README MUSDB :", missing)

    durations = {s: np.array([sf.info(f"{d}/mixture.wav").duration for d in dirs[s]]) / 60 for s in subsets}
    for s in subsets:
        print(f"{s:5s} {len(names[s]):3d} morceaux, {durations[s].sum():5.1f} min au total, médiane {np.median(durations[s]):.2f} min")

    # --- genres et origines
    genres = sorted({meta[n]["genre"] for s in subsets for n in names[s]}, key=lambda g: -sum(meta[n]["genre"] == g for s in subsets for n in names[s]))
    origins = sorted({meta[n]["origin"] for s in subsets for n in names[s]})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1.6, 1.6]}, constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(genres))
    left = np.zeros(len(genres))
    for s in subsets:
        c = np.array([sum(meta[n]["genre"] == g for n in names[s]) for g in genres])
        ax.barh(y, c, left=left, color=colors[s], label=f"{s} ({len(names[s])})", edgecolor="white", linewidth=1.5)
        left += c
    ax.set_yticks(y, genres)
    ax.invert_yaxis()
    for yi, tot in zip(y, left):
        ax.text(tot + 0.5, yi, int(tot), va="center", color=INK_2, fontsize=9)
    ax.set_title("Genre")
    ax.set_xlabel("morceaux")
    ax.legend(loc="center right")

    ax = axes[1]
    left = np.zeros(len(origins))
    y = np.arange(len(origins))
    for s in subsets:
        c = np.array([sum(meta[n]["origin"] == o for n in names[s]) for o in origins])
        ax.barh(y, c, left=left, color=colors[s], edgecolor="white", linewidth=1.5)
        left += c
    ax.set_yticks(y, [SHORT_ORIGIN.get(o, o) for o in origins])
    ax.invert_yaxis()
    ax.set_title("Origine")
    ax.set_xlabel("morceaux")

    ax = axes[2]
    bins = np.linspace(0, max(d.max() for d in durations.values()) + 0.5, 14)
    for s in subsets:
        ax.hist(durations[s], bins=bins, histtype="step", linewidth=2, color=colors[s], label=s, density=True)
    ax.set_title("Durée des morceaux")
    ax.set_xlabel("minutes")
    ax.set_ylabel("densité")
    fig.suptitle("MUSDB18 : répartition des morceaux dans les sous-ensembles", fontweight="bold")
    save(fig, "dataset_distribution.png")

    # --- artistes partagés entre sous-ensembles (fuite potentielle, B6 du walkthrough)
    art = {s: Counter(artist(n) for n in names[s]) for s in subsets}
    shared_train_test = sorted(set(art["train"]) & set(art["test"]))
    shared_train_val = sorted(set(art["train"]) & set(art["val"]))
    shared_val_test = sorted(set(art["val"]) & set(art["test"]))
    print("Artistes train/test partagés :", shared_train_test)
    print("Artistes train/val partagés :", shared_train_val)
    print("Artistes val/test partagés :", shared_val_test)
    n_test_leak = sum(art["test"][a] for a in shared_train_test)
    print(f"Morceaux de test dont l'artiste est dans le train : {n_test_leak}/{len(names['test'])}")

    allart = Counter()
    for s in subsets:
        allart.update(art[s])
    top = [a for a, _ in allart.most_common(12)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for s in subsets:
        c = np.array([art[s][a] for a in top])
        ax.barh(y, c, left=left, color=colors[s], label=s, edgecolor="white", linewidth=1.5)
        left += c
    ax.set_yticks(y, top)
    ax.invert_yaxis()
    ax.set_xlabel("morceaux")
    ax.set_title(f"Artistes les plus représentés ({len(allart)} artistes au total)")
    ax.legend(loc="lower right")
    save(fig, "dataset_artists.png")

    # --- tableau genre x sous-ensemble en pourcentage, pour le README
    print("\nGenre            " + "  ".join(f"{s:>6s}" for s in subsets))
    for g in genres:
        row = [100 * sum(meta[n]["genre"] == g for n in names[s]) / len(names[s]) for s in subsets]
        print(f"{g:16s} " + "  ".join(f"{v:5.1f}%" for v in row))


if __name__ == "__main__":
    main()
