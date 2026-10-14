"""Style, palette et lecture des métadonnées partagés par les scripts d'analyse."""
import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
MUSDB = os.path.join(ROOT, "musdb18")
PROCESSED = os.path.join(MUSDB, "processed")

# 3 premiers slots catégoriels de la palette de référence (valident aussi en all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "legend.frameon": False,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"-> {os.path.relpath(path, ROOT)}")


def artist(track_name):
    return track_name.split(" - ")[0]


# Noms différents entre musdb18/README.md et les dossiers (typos du README MUSDB), et morceau absent du README.
ALIASES = {"Patrick Talbot - Set Me Free": "Patrick Talbot - Set Free Me"}
UNKNOWN = {"genre": "Inconnu", "origin": "Inconnu"}


class Metadata(dict):
    """Dictionnaire {titre: {genre, origin}} tolérant aux typos : clé normalisée, alias, valeur « Inconnu » sinon."""

    @staticmethod
    def _key(name):
        return re.sub(r"[^a-z0-9]", "", ALIASES.get(name, name).lower())

    def __getitem__(self, name):
        return super().get(self._key(name), UNKNOWN)

    def __contains__(self, name):
        return True


def read_metadata():
    """{titre: {genre, origin}} lu dans musdb18/README.md (lignes « - Artiste - Titre,Genre,Origine,Licence »)."""
    meta = Metadata()
    with open(os.path.join(MUSDB, "README.md")) as f:
        for line in f:
            m = re.match(r"^- (.+),([^,]+),(DSD|MedleyDB|NI|heise|[^,]+),([^,]+)$", line.strip())
            if m:
                dict.__setitem__(meta, Metadata._key(m.group(1)), {"genre": m.group(2), "origin": m.group(3)})
    return meta


def track_dirs():
    """{'train': [...], 'val': [...], 'test': [...]} selon musdb18/split.json (val = sous-ensemble du train officiel)."""
    with open(os.path.join(MUSDB, "split.json")) as f:
        split = json.load(f)
    test = sorted(os.listdir(os.path.join(PROCESSED, "test")))
    return {
        "train": [os.path.join(ROOT, p) for p in split["train"]],
        "val": [os.path.join(ROOT, p) for p in split["val"]],
        "test": [os.path.join(PROCESSED, "test", t) for t in test],
    }


def name_of(path):
    return os.path.basename(path.rstrip("/"))


def bootstrap_ci(values, stat=np.median, n=10_000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    s = stat(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)
