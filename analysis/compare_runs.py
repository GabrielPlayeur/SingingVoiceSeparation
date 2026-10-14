"""Compare deux runs évalués sur les mêmes 50 morceaux de test (JSON de `src.evaluate -o`), par test apparié.

Usage :
    python -m analysis.compare_runs outputs/oracle_baseline.json outputs/stereo.json --names mono stereo
Sorties : figures/compare_<a>_<b>[_<metrique>].png + résumé texte (différence médiane, IC95 % bootstrap, Wilcoxon, nombre de victoires).

Un gain n'est à retenir que s'il dépasse la variabilité entre graines (ROADMAP, étape 0) : ce script ne mesure que le bruit
entre morceaux. Avec une seule graine par configuration, la conclusion reste à confirmer.
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *


def load(path):
    with open(path) as f:
        return {r["track"]: r for r in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--metric", default="sdr", help="clé du JSON : sdr, si_sdri (protocole natif du modèle), ou à protocole\n"
                    "égal : sdr_mono, sdr_stereo, si_sdri_mono, si_sdri_stereo")
    ap.add_argument("--exclude", nargs="*", default=[], help="sous-chaînes de noms de morceaux à écarter (ex. 'PR -')")
    args = ap.parse_args()
    setup_style()
    na, nb = args.names

    A, B = load(args.a), load(args.b)
    tracks = sorted(t for t in set(A) & set(B) if not any(x in t for x in args.exclude))
    print(f"{len(tracks)} morceaux communs ({len(A)} / {len(B)})")
    a = np.array([A[t][args.metric] for t in tracks])
    b = np.array([B[t][args.metric] for t in tracks])
    d = b - a
    unit = "dB"
    for name, v in ((na, a), (nb, b)):
        m, lo, hi = bootstrap_ci(v)
        print(f"{args.metric} {name:10s} médiane {m:.2f} {unit}  IC95% [{lo:.2f}, {hi:.2f}]  moyenne {np.nanmean(v):.2f}")
    m, lo, hi = bootstrap_ci(d)
    w = stats.wilcoxon(a, b, nan_policy="omit")
    print(f"Différence ({nb} − {na}) : médiane {m:+.2f} {unit}  IC95% bootstrap [{lo:+.2f}, {hi:+.2f}]  moyenne {np.nanmean(d):+.2f}")
    print(f"{nb} meilleur sur {(d > 0).sum()}/{len(d)} morceaux ; Wilcoxon apparié p = {w.pvalue:.3g}")
    order = np.argsort(d)
    print("Plus fortes baisses :", [(tracks[i], round(float(d[i]), 1)) for i in order[:3]])
    print("Plus fortes hausses :", [(tracks[i], round(float(d[i]), 1)) for i in order[-3:]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.scatter(a, b, color=BLUE, s=34, edgecolor="white", linewidth=0.8, zorder=3)
    lim = [min(np.nanmin(a), np.nanmin(b)) - 1, max(np.nanmax(a), np.nanmax(b)) + 1]
    ax.plot(lim, lim, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{args.metric} {na} (dB)"); ax.set_ylabel(f"{args.metric} {nb} (dB)")
    ax.set_title("Par morceau (au-dessus de la diagonale : " + nb + " meilleur)")

    ax = axes[1]
    for x, y in zip(a, b):
        ax.plot([0, 1], [x, y], color=BLUE if y >= x else ORANGE, alpha=0.5, linewidth=1)
    ax.scatter(np.zeros(len(a)), a, color=INK_2, s=12, zorder=3); ax.scatter(np.ones(len(b)), b, color=INK_2, s=12, zorder=3)
    ax.plot([0, 1], [np.nanmedian(a), np.nanmedian(b)], color=INK, linewidth=3)
    ax.set_xticks([0, 1], [na, nb]); ax.set_xlim(-0.3, 1.3); ax.set_ylabel("dB")
    ax.set_title("Même morceau, deux modèles (gras = médiane)")

    ax = axes[2]
    ax.hist(d, bins=20, color=BLUE, edgecolor="white")
    ax.axvline(0, color=INK_2, linewidth=1, linestyle="--")
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.25, label=f"IC95 % de la médiane [{lo:+.2f}, {hi:+.2f}]")
    ax.axvline(m, color=ORANGE, linewidth=2)
    ax.set_xlabel(f"{nb} − {na} (dB)"); ax.set_ylabel("morceaux"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Différence apparié (Wilcoxon p = {w.pvalue:.2g})")
    fig.suptitle(f"{na} vs {nb} sur {len(tracks)} morceaux de test", fontweight="bold")
    save(fig, f"compare_{na}_{nb}{'' if args.metric == 'sdr' else '_' + args.metric}{'_sans_exclus' if args.exclude else ''}.png")


if __name__ == "__main__":
    main()
