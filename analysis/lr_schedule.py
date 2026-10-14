"""Comparaison des runs d'optimisation de la baseline (learning rate, scheduler) à partir de l'historique W&B.

Critère de choix (sur la VALIDATION, jamais sur le test) : moyenne de `val/si_sdr_median` sur les 10 DERNIÈRES époques. Le maximum
(utilisé pour le checkpoint) récompense une époque chanceuse quand la courbe oscille ; la moyenne finale mesure ce que le run
délivre réellement. L'écart-type sur ces 10 époques donne l'ordre de grandeur du bruit : en dessous, deux runs ne se
distinguent pas (une seule graine par run).

Usage : python -m analysis.lr_schedule --run stereo=fgvzjxua --run lr3e4=<id W&B> --run cosine=<id> --run lr3e4_cosine=<id>
L'id est celui du dossier outputs/VoiceSeparation/<id>/ (ou de l'URL du run W&B). Sans --run : la baseline seule.
Sorties : figures/lr_schedule.png, figures/lr_schedule.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from ._common import *

PROJECT = "playeur-audio-team/VoiceSeparation"
METRIC = "val/si_sdr_median"
LAST = 10  # nombre d'époques du critère de choix
COLORS = [BLUE, ORANGE, AQUA, INK_2, "#8a5fd1"]


def load_history(run_id):
    import wandb
    run = wandb.Api(timeout=30).run(f"{PROJECT}/{run_id}")
    h = run.history(keys=[METRIC, "epoch"], pandas=True, samples=5000).dropna(subset=[METRIC])
    h = h.drop_duplicates("epoch").sort_values("epoch")
    return run.name, h["epoch"].to_numpy(), h[METRIC].to_numpy()


def summarize(epochs, y):
    tail = y[-LAST:]
    return dict(epochs=int(len(y)), best=float(y.max()), best_epoch=int(epochs[y.argmax()]),
                last_mean=float(tail.mean()), last_std=float(tail.std(ddof=1)), final=float(y[-1]),
                epoch_to_epoch_std=float(np.diff(y).std(ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="NOM=ID_WANDB")
    args = ap.parse_args()
    setup_style()
    runs = dict(r.split("=", 1) for r in args.run) or {"stereo": "fgvzjxua"}

    results, curves = {}, {}
    for name, run_id in runs.items():
        wandb_name, epochs, y = load_history(run_id)
        results[name] = summarize(epochs, y) | {"wandb_run": run_id}
        curves[name] = (epochs, y)
        r = results[name]
        print(f"{name:14s} ({wandb_name}) {r['epochs']:3d} époques | meilleur {r['best']:.2f} dB (époque {r['best_epoch']}) | "
              f"moyenne des {LAST} dernières {r['last_mean']:.2f} ± {r['last_std']:.2f} dB | bruit époque-à-époque {r['epoch_to_epoch_std']:.2f} dB")
    if len(results) > 1:
        ranked = sorted(results, key=lambda n: -results[n]["last_mean"])
        best = ranked[0]
        print(f"\nCritère (moyenne des {LAST} dernières époques) : " + " > ".join(f"{n} {results[n]['last_mean']:.2f}" for n in ranked))
        for n in ranked[1:]:
            gap = results[best]["last_mean"] - results[n]["last_mean"]
            noise = max(results[best]["last_std"], results[n]["last_std"])
            print(f"  {best} - {n} : {gap:.2f} dB (bruit {noise:.2f} dB) -> {'écart au-dessus du bruit' if gap > noise else 'non distinguable'}")

    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    for i, (name, (epochs, y)) in enumerate(curves.items()):
        c = COLORS[i % len(COLORS)]
        ax.plot(epochs, y, color=c, alpha=0.25, linewidth=1)
        k = min(LAST, len(y))
        smooth = np.convolve(y, np.ones(k) / k, mode="valid")  # moyenne glissante sur LAST époques
        ax.plot(epochs[k - 1:], smooth, color=c, linewidth=2, label=f"{name} (fin : {results[name]['last_mean']:.2f} dB)")
    ax.set_xlabel("époque")
    ax.set_ylabel("val/si_sdr_median (dB)")
    ax.set_title(f"Optimisation de la baseline (trait fin : valeur par époque, trait épais : moyenne glissante sur {LAST})")
    ax.legend(loc="lower right")
    save(fig, "lr_schedule.png")
    with open(os.path.join(FIG_DIR, "lr_schedule.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
