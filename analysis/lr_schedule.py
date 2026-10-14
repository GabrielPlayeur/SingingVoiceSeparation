"""Comparaison des runs d'optimisation de la baseline (learning rate, scheduler) à partir de l'historique W&B.

Critère de choix (sur la VALIDATION, jamais sur le test) : moyenne de `val/si_sdr_median` sur les 10 DERNIÈRES époques. Le maximum
(utilisé pour le checkpoint) récompense une époque chanceuse quand la courbe oscille ; la moyenne finale mesure ce que le run
délivre réellement. L'écart-type sur ces 10 époques donne l'ordre de grandeur du bruit : en dessous, deux runs ne se
distinguent pas (une seule graine par run).

Usage : python -m analysis.lr_schedule --run stereo=fgvzjxua --run lr3e4=<id W&B> --run cosine=<id> --run lr3e4_cosine=<id>
L'id est celui du dossier outputs/VoiceSeparation/<id>/ (ou de l'URL du run W&B). Sans --run : la baseline seule.
Un run du serveur jamais synchronisé (W&B hors ligne) se lit directement depuis son fichier : --run lr3e4=chemin/run-<id>.wandb
(lecture locale du journal binaire W&B, rien n'est envoyé).
Sorties : figures/lr_schedule.png, figures/lr_schedule.json
"""
import argparse
import json
import os
import struct

import numpy as np
import matplotlib.pyplot as plt

from ._common import *

PROJECT = "playeur-audio-team/VoiceSeparation"
METRIC = "val/si_sdr_median"
LAST = 10  # nombre d'époques du critère de choix
COLORS = [BLUE, ORANGE, AQUA, INK_2, "#8a5fd1"]


def _wandb_records(path):
    """Enregistrements protobuf d'un journal W&B hors ligne (format LevelDB : en-tête de 7 octets, blocs de 32 768 octets,
    chaque enregistrement précédé de crc(4) + longueur(2) + type(1) ; 1 = complet, 2/3/4 = début/milieu/fin)."""
    data = open(path, "rb").read()
    block, pos, buf = 32768, 7, b""
    while pos < len(data):
        if block - pos % block < 7:
            pos += block - pos % block
            continue
        _, length, kind = struct.unpack("<IHB", data[pos:pos + 7])
        pos += 7
        if kind == 0 and length == 0:
            pos += (block - pos % block) % block
            continue
        chunk = data[pos:pos + length]
        pos += length
        if kind == 1:
            yield chunk
        elif kind == 2:
            buf = chunk
        elif kind == 3:
            buf += chunk
        elif kind == 4:
            yield buf + chunk
            buf = b""


def load_offline(path):
    from wandb.proto import wandb_internal_pb2 as pb
    rows = {}
    for raw in _wandb_records(path):
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") == "history":
            h = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in rec.history.item}
            if METRIC in h and "epoch" in h:
                rows[int(h["epoch"])] = h[METRIC]
    epochs = sorted(rows)
    return os.path.basename(path), np.array(epochs), np.array([rows[e] for e in epochs])


def load_history(run_id):
    if run_id.endswith(".wandb"):
        return load_offline(run_id)
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

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]})
    for i, (name, (epochs, y)) in enumerate(curves.items()):
        c = COLORS[i % len(COLORS)]
        ax.plot(epochs, y, color=c, alpha=0.25, linewidth=1)
        k = min(LAST, len(y))
        smooth = np.convolve(y, np.ones(k) / k, mode="valid")  # moyenne glissante sur LAST époques
        ax.plot(epochs[k - 1:], smooth, color=c, linewidth=2, label=f"{name} (fin : {results[name]['last_mean']:.2f} dB)")
    ax.set_xlabel("époque")
    ax.set_ylabel("val/si_sdr_median (dB)")
    ax.set_title(f"Validation par époque (trait fin : brut, épais : moyenne glissante sur {LAST})")
    ax.set_ylim(2.0, 4.3)  # les premières époques (de -4 à 2 dB) écrasent l’échelle ; la comparaison se joue au-dessus de 2 dB
    ax.legend(loc="lower right")

    order = sorted(results, key=lambda n: results[n]["last_mean"])
    pos = np.arange(len(order))
    bx.barh(pos, [results[n]["last_mean"] for n in order], xerr=[results[n]["last_std"] for n in order],
            color=[COLORS[list(curves).index(n) % len(COLORS)] for n in order], error_kw={"ecolor": INK_2, "capsize": 3})
    for p_, n in zip(pos, order):
        bx.text(results[n]["last_mean"] + results[n]["last_std"] + 0.03, p_, f"{results[n]['last_mean']:.2f}", va="center", fontsize=9)
    bx.set_yticks(pos, order)
    bx.set_xlim(min(results[n]["last_mean"] for n in order) - 0.4, max(results[n]["last_mean"] for n in order) + 0.35)
    bx.set_xlabel(f"moyenne des {LAST} dernières époques ± écart-type (dB)")
    bx.set_title("Critère de choix")
    bx.grid(axis="y", visible=False)
    save(fig, "lr_schedule.png")
    with open(os.path.join(FIG_DIR, "lr_schedule.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
