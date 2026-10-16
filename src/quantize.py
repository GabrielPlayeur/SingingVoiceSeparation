"""Quantification int8 d'un checkpoint : PTQ, ablations, sensibilité par couche, QAT, export réel, évaluation (experiments/08).

Tout tourne en local (GPU 4 Go pour la simulation et le QAT, CPU pour l'int8 réel). Sous-commandes :

    python -m src.quantize ptq CKPT            # un réglage PTQ simulé : SI-SDR de validation fp32 vs quantifié
    python -m src.quantize sweep CKPT          # grille d'ablations PTQ (largeur de bits, plage, granularité...) -> JSON
    python -m src.quantize sensitivity CKPT    # un quantificateur à la fois (Nagel § 3.7) -> JSON
    python -m src.quantize ranges CKPT         # plage observée vs quantiles de chaque activation -> JSON
    python -m src.quantize qat CKPT            # fine-tuning avec quantification fictive -> checkpoint .qat.pt
    python -m src.quantize export CKPT         # vrai int8 (CPU) : taille, latence, accord avec la simulation
    python -m src.quantize evaluate CKPT       # SDR museval sur les morceaux de test avec le modèle simulé ou réel

Les métriques rapides (SI-SDR médian de validation) servent aux ablations ; la décision finale se lit sur le SDR de test.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from .data.data import MUSDB18DataModule
from .module import VoiceSeparationModule
from .quantization import (QuantConfig, SimUNet, calibrate, export_int8, latency_ms, state_dict_bytes)
from .utils.utils import si_sdr

CONFIG_DATA = os.path.join(os.path.dirname(__file__), "configs", "data", "musdb.yaml")


# ----------------------------------------------------------------------------------------------------------------------
# Modèle et données
# ----------------------------------------------------------------------------------------------------------------------

def load_module(ckpt, device):
    module = VoiceSeparationModule.load_from_checkpoint(ckpt, map_location="cpu").eval()
    return module.to(device)


def make_datamodule(args, stage="fit", val_crops=None):
    """`val_crops` : extraits de validation par morceau (15 par défaut = ceux de l'entraînement, ~1 Go en mémoire ; 1 quand on n'en a pas
    besoin, par exemple pour l'évaluation sur le jeu de test)."""
    cfg = OmegaConf.load(CONFIG_DATA)
    cfg.dataset_path, cfg.split_path = args.dataset_path, args.split_path
    if val_crops is not None:
        cfg.val_crops_per_track = val_crops
    cfg.channels, cfg.num_workers, cfg.batch_size = 2, args.num_workers, args.batch_size
    dm = MUSDB18DataModule(**cfg)
    dm.setup(stage)
    return dm


def val_loader(dm):
    """Extraits de validation déjà en mémoire : pas de processus de lecture (un par worker copierait ~0,5 Go chacun, la RAM de ce PC est de 8 Go)."""
    from torch.utils.data import DataLoader
    return DataLoader(dm.val_dataset, batch_size=dm.hparams.batch_size, shuffle=False, num_workers=0)


def train_loader(dm, num_workers):
    """Extraits d'entraînement. Processus « spawn » (pas « fork ») : un worker forké hérite des ~0,5 Go d'extraits de validation en mémoire
    et les recopie (compteurs de références), ce qui fait déborder la RAM de 8 Go de ce PC dès qu'un autre calcul tourne."""
    from torch.utils.data import DataLoader
    return DataLoader(dm.train_dataset, batch_size=dm.hparams.batch_size, shuffle=True, num_workers=num_workers,
                      persistent_workers=num_workers > 0, multiprocessing_context="spawn" if num_workers > 0 else None)


def network_inputs(module, batch, device):
    """Entrée du U-Net (amplitude transformée) + ce qu'il faut pour reconstruire la forme d'onde, comme `validation_step`."""
    batch = {k: v.to(device) for k, v in batch.items()}
    batch = module.on_after_batch_transfer(batch, 0)
    return batch


@torch.no_grad()
def val_sisdr(module, loader, device, max_batches=None):
    """SI-SDR de validation (médiane, moyenne) sur des extraits fixes : la même quantité que `val/si_sdr_median` de l'entraînement."""
    module.eval()
    out = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        b = network_inputs(module, batch, device)
        mask = module(b["mix_amp"])
        est = module._to_wave(mask * b["mix_amp"], b["mix_phase"], b["vocals"].shape[-1])
        out.append(si_sdr(est, b["vocals"]).cpu())
    s = torch.cat(out)
    return {"median": float(torch.quantile(s, 0.5)), "mean": float(s.mean()), "n": int(s.numel())}


def calibration_batches(module, dm, device, n_crops=64, batch_size=8, seed=0):
    """`n_crops` extraits d'ENTRAÎNEMENT tirés au hasard (graine fixe) -> entrées du réseau. Pas de morceau de validation ni de test
    (Nagel § 3.5 : un petit jeu de calibration non étiqueté suffit ; ~100 lots chez Krishnamoorthi § 3.1.2)."""
    np.random.seed(seed)
    ds = dm.train_dataset
    crops = [ds[i] for i in range(n_crops)]
    out = []
    for i in range(0, n_crops, batch_size):
        batch = {k: torch.stack([c[k] for c in crops[i:i + batch_size]]) for k in crops[0]}
        b = network_inputs(module, batch, device)
        out.append(module_input(module, b["mix_amp"]))
    return out


def module_input(module, mix_amp):
    from .utils.input_transform import transform_amplitude
    return transform_amplitude(mix_amp, module.hparams.input_transform, module.hparams.input_power)


def attach(module, sim):
    """Remplace le U-Net du module par la version simulée : `module(...)`, `_to_wave`, `evaluate` s'utilisent ensuite tels quels."""
    q = VoiceSeparationModule(**module.hparams)  # même hyper-paramètres, même fenêtre STFT ; le module d'origine reste intact
    q.model = sim
    try:
        device = next(sim.parameters()).device
    except StopIteration:   # modèle int8 réel : poids empaquetés, pas de paramètres ; il ne tourne que sur CPU
        device = torch.device("cpu")
    return q.to(device).eval()


def build_sim(module, cfg, dm, device, n_calib=64):
    sim = SimUNet(module.model, cfg).to(device)
    calibrate(sim, calibration_batches(module, dm, device, n_calib), quantize_weights=cfg.quantize_weights, quantize_acts=cfg.quantize_acts)
    return sim


def cfg_from_args(args):
    return QuantConfig(w_bits=args.w_bits, a_bits=args.a_bits, w_per_channel=not args.w_per_tensor, convt_per_channel=args.convt_per_channel,
                       w_range=args.w_range, act_range=args.act_range, percentile=args.percentile, reduce_range=not args.no_reduce_range,
                       quantize_input=not args.fp_input, quantize_weights=not args.no_weights, quantize_acts=not args.no_acts)


# ----------------------------------------------------------------------------------------------------------------------
# Sous-commandes
# ----------------------------------------------------------------------------------------------------------------------

def cmd_ptq(args, device):
    module = load_module(args.ckpt, device)
    dm = make_datamodule(args)
    loader = val_loader(dm)
    ref = val_sisdr(module, loader, device)
    cfg = cfg_from_args(args)
    sim = build_sim(module, cfg, dm, device, args.calib_crops)
    q = val_sisdr(attach(module, sim), loader, device)
    print(f"{cfg}\nfp32  val SI-SDR médian {ref['median']:.3f} dB (n={ref['n']})\nquant val SI-SDR médian {q['median']:.3f} dB  "
          f"(écart {q['median'] - ref['median']:+.3f} dB)")


SWEEP = [  # (nom, surcharges de QuantConfig) : une variable à la fois autour du réglage de base W8A8 minmax
    ("fp32 (contrôle : quantificateurs coupés)", dict(quantize_weights=False, quantize_acts=False)),
    ("poids seuls W8 (par canal)", dict(quantize_acts=False)),
    ("poids seuls W8 (par tenseur partout)", dict(quantize_acts=False, w_per_channel=False, convt_per_channel=False)),
    ("poids seuls W8 (ConvT par canal)", dict(quantize_acts=False, convt_per_channel=True)),
    ("poids seuls W8 (plage MSE)", dict(quantize_acts=False, w_range="mse")),
    ("poids seuls W6", dict(quantize_acts=False, w_bits=6)),
    ("poids seuls W4", dict(quantize_acts=False, w_bits=4)),
    ("activations seules A8 (minmax)", dict(quantize_weights=False)),
    ("activations seules A8 (percentile 99.99)", dict(quantize_weights=False, act_range="percentile")),
    ("activations seules A8 (percentile 99.9)", dict(quantize_weights=False, act_range="percentile", percentile=99.9)),
    ("activations seules A8 (mse)", dict(quantize_weights=False, act_range="mse")),
    ("activations seules A8, 8 bits pleins (sans reduce_range)", dict(quantize_weights=False, reduce_range=False)),
    ("activations seules A8, entrée en flottant", dict(quantize_weights=False, quantize_input=False)),
    ("activations seules A8 (percentile 99.99), 8 bits pleins", dict(quantize_weights=False, act_range="percentile", reduce_range=False)),
    ("W8A8 minmax (réglage par défaut)", dict()),
    ("W8A8 minmax, 8 bits pleins", dict(reduce_range=False)),
    ("W8A8 percentile 99.99, 8 bits pleins", dict(act_range="percentile", reduce_range=False)),
    ("W8A8 percentile 99.99", dict(act_range="percentile")),
    ("W8A8 mse (poids et activations)", dict(act_range="mse", w_range="mse")),
    ("W8A8 mse, entrée en flottant", dict(act_range="mse", w_range="mse", quantize_input=False)),
    ("W8A8 percentile 99.99, entrée en flottant", dict(act_range="percentile", quantize_input=False)),
]


def cmd_sweep(args, device):
    module = load_module(args.ckpt, device)
    dm = make_datamodule(args)
    loader = val_loader(dm)
    batches = calibration_batches(module, dm, device, args.calib_crops)
    ref = val_sisdr(module, loader, device)
    print(f"fp32 : val SI-SDR médian {ref['median']:.3f} dB (n={ref['n']})")
    rows = []
    for name, over in SWEEP:
        cfg = QuantConfig(**over)
        sim = SimUNet(module.model, cfg).to(device)
        calibrate(sim, batches, quantize_weights=cfg.quantize_weights, quantize_acts=cfg.quantize_acts)
        r = val_sisdr(attach(module, sim), loader, device)
        rows.append({"name": name, "config": vars(cfg), "median": r["median"], "mean": r["mean"], "delta_median": r["median"] - ref["median"]})
        print(f"{name:62s} {r['median']:7.3f} dB   {r['median'] - ref['median']:+.3f}", flush=True)
    save_json(args.out or "figures/quant_sweep.json", {"ckpt": args.ckpt, "fp32": ref, "rows": rows, "calib_crops": args.calib_crops})


def cmd_sensitivity(args, device):
    """Nagel § 3.7 « per-layer analysis » : on quantifie UN seul quantificateur (les autres restent en flottant) et on mesure la chute."""
    module = load_module(args.ckpt, device)
    dm = make_datamodule(args)
    loader = val_loader(dm)
    ref = val_sisdr(module, loader, device, args.max_batches)
    cfg = cfg_from_args(args)
    sim = SimUNet(module.model, cfg).to(device)
    calibrate(sim, calibration_batches(module, dm, device, args.calib_crops))
    q = attach(module, sim)
    rows = []
    print(f"fp32 : {ref['median']:.3f} dB ; {cfg}")
    for kind, names in (("activation", sim.act_names), ("poids", sim.w_names)):
        for n in names:
            if kind == "activation":
                sim.set_mode(act="quant", weights="off", only_act=[n])
            else:
                sim.set_mode(act="off", weights="quant", only_w=[n])
            r = val_sisdr(q, loader, device, args.max_batches)
            rows.append({"kind": kind, "name": n, "median": r["median"], "delta_median": r["median"] - ref["median"]})
            print(f"{kind:10s} {n:12s} {r['median'] - ref['median']:+.3f} dB", flush=True)
    save_json(args.out or "figures/quant_sensitivity.json", {"ckpt": args.ckpt, "fp32": ref, "config": vars(cfg), "rows": rows})


@torch.no_grad()
def cmd_ranges(args, device):
    """Plage observée de chaque activation (min, max) contre ses quantiles 0,01 % et 99,99 % : explique pourquoi la plage min-max est
    mauvaise (un pas de quantification fixé par quelques valeurs extrêmes)."""
    module = load_module(args.ckpt, device)
    dm = make_datamodule(args)
    sim = SimUNet(module.model, cfg_from_args(args)).to(device).eval()
    for n in sim.act_names:
        sim.Q(n).reset()
    sim.set_mode(act="observe", weights="off")
    with torch.random.fork_rng():
        torch.manual_seed(0)
        for x in calibration_batches(module, dm, device, args.calib_crops):
            sim(x)
    rows = []
    print(f"{'quantificateur':14s} {'min':>9s} {'p0.01':>9s} {'p99.99':>9s} {'max':>9s} {'max/p99.99':>11s} {'pas 8 bits (minmax)':>20s} {'pas (p99.99)':>13s}")
    for n in sim.act_names:
        qz = sim.Q(n)
        smp = torch.cat(qz._samples)
        lo, hi = float(qz.observer.min_val), float(qz.observer.max_val)
        q = torch.quantile(smp, torch.tensor([1e-4, 0.9999]))
        step_mm = (hi - lo) / qz.qmax
        step_p = (max(float(q[1]), 0) - min(float(q[0]), 0)) / qz.qmax
        rows.append({"name": n, "min": lo, "max": hi, "p0.01": float(q[0]), "p99.99": float(q[1]), "step_minmax": step_mm, "step_p9999": step_p})
        print(f"{n:14s} {lo:9.3f} {float(q[0]):9.3f} {float(q[1]):9.3f} {hi:9.3f} {hi / max(float(q[1]), 1e-9):11.1f} {step_mm:20.4f} {step_p:13.4f}")
    save_json(args.out or "figures/quant_ranges.json", {"ckpt": args.ckpt, "calib_crops": args.calib_crops, "rows": rows})


def cmd_qat(args, device):
    module = load_module(args.ckpt, device)
    dm = make_datamodule(args)
    cfg = cfg_from_args(args)
    loader_val = val_loader(dm)
    ref = val_sisdr(module, loader_val, device)
    sim = build_sim(module, cfg, dm, device, args.calib_crops)
    q = attach(module, sim)
    ptq = val_sisdr(q, loader_val, device)
    print(f"fp32 {ref['median']:.3f} dB | PTQ {ptq['median']:.3f} dB ({ptq['median'] - ref['median']:+.3f}) | QAT : {args.epochs} époques, lr {args.lr}{' cosinus' if args.cosine else ''}", flush=True)
    params = [p for p in sim.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    steps_per_epoch = len(range(0, dm.hparams.epoch_size, dm.hparams.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=0.0) if args.cosine else None
    # quant : plages d'activation figées à la calibration (percentile...) ; qat : moyenne mobile min/max ; off : témoin sans quantification
    act_mode = ("qat" if args.update_ranges else "quant") if cfg.quantize_acts else "off"
    w_mode = "qat" if cfg.quantize_weights else "off"
    sim.set_mode(act=act_mode, weights=w_mode)
    best, best_state, history = ptq["median"], None, [{"epoch": 0, "val_median": ptq["median"]}]
    loader = train_loader(dm, args.num_workers)
    for epoch in range(1, args.epochs + 1):
        sim.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in loader:
            b = network_inputs(module, batch, device)
            mask = q(b["mix_amp"])
            loss = q._loss(b, mask * b["mix_amp"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tot += float(loss.detach()); n += 1
        sim.eval()
        sim.set_mode(act="quant" if cfg.quantize_acts else "off", weights="quant" if cfg.quantize_weights else "off")
        sim.finalize_weights()
        v = val_sisdr(q, loader_val, device)
        sim.set_mode(act=act_mode, weights=w_mode)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_median": v["median"]})
        print(f"époque {epoch:2d}  loss {tot / n:.4f}  val SI-SDR {v['median']:.3f} dB ({v['median'] - ref['median']:+.3f} vs fp32)  {time.time() - t0:.0f} s", flush=True)
        if args.select == "best" and v["median"] > best or args.select == "last" and epoch == args.epochs:
            best, best_state = v["median"], {k: x.detach().cpu().clone() for k, x in sim.state_dict().items()}
    if best_state is None:
        print("QAT n'a pas dépassé le PTQ de départ : on garde l'état PTQ (aucun checkpoint écrit).")
        return
    sim.load_state_dict(best_state)
    sim.eval(); sim.set_mode(act="quant" if cfg.quantize_acts else "off", weights="quant" if cfg.quantize_weights else "off"); sim.finalize_weights()
    out = args.out or args.ckpt.replace(".ckpt", "") + f".qat_{args.tag}.pt"
    torch.save({"state_dict": sim.state_dict(), "config": vars(cfg), "ckpt": args.ckpt, "history": history, "fp32_val": ref, "ptq_val": ptq,
                "epochs": args.epochs, "lr": args.lr, "cosine": args.cosine}, out)
    print(f"état retenu ({args.select}) : {best:.3f} dB -> {out}")


def load_sim(module, path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    sim = SimUNet(module.model, QuantConfig(**d["config"])).to(device)
    sim.load_state_dict(d["state_dict"])
    sim.eval()
    c = sim.cfg
    sim.set_mode(act="quant" if c.quantize_acts else "off", weights="quant" if c.quantize_weights else "off")
    sim.finalize_weights()
    return sim, d


def cmd_export(args, device):
    cpu = torch.device("cpu")
    module = load_module(args.ckpt, cpu)
    dm = make_datamodule(args, val_crops=args.val_crops)
    if args.qat:
        sim, _ = load_sim(module, args.qat, cpu)
    else:
        sim = build_sim(module, cfg_from_args(args), dm, cpu, args.calib_crops)
    real = export_int8(sim)
    loader = val_loader(dm)
    n = args.max_batches
    r_fp, r_sim = val_sisdr(module, loader, cpu, n), val_sisdr(attach(module, sim), loader, cpu, n)
    r_real = val_sisdr(attach(module, real), loader, cpu, n)
    print(f"val SI-SDR médian : fp32 {r_fp['median']:.3f} | simulation {r_sim['median']:.3f} | int8 réel {r_real['median']:.3f} dB (n={r_fp['n']})")
    x = next(iter(val_loader(dm)))
    b = network_inputs(module, {k: v[:1] for k, v in x.items()}, cpu)
    inp = module_input(module, b["mix_amp"])
    with torch.no_grad():
        m_fp, m_sim, m_real = module.model(inp), sim(inp), real(inp)
    print(f"masque : |sim - réel| moyen {float((m_sim - m_real).abs().mean()):.5f} (max {float((m_sim - m_real).abs().max()):.3f}) ; "
          f"|fp32 - réel| moyen {float((m_fp - m_real).abs().mean()):.5f}")
    fp_bytes, int8_bytes = state_dict_bytes(module.model), state_dict_bytes(real)
    print(f"taille : fp32 {fp_bytes / 1e6:.2f} Mo (BatchNorm comprise) | int8 {int8_bytes / 1e6:.2f} Mo ({fp_bytes / int8_bytes:.2f}x)")
    lat = {}
    for threads in args.threads:
        lat[threads] = {"fp32": latency_ms(module.model, inp, threads), "int8": latency_ms(real, inp, threads)}
        f, i = lat[threads]["fp32"][0], lat[threads]["int8"][0]
        print(f"latence 1 extrait 5,93 s, {threads} thread(s) : fp32 {f:.0f} ms | int8 {i:.0f} ms ({f / i:.2f}x)")
    save_json(args.out or "figures/quant_export.json", {"ckpt": args.ckpt, "qat": args.qat, "val": {"fp32": r_fp, "sim": r_sim, "real": r_real},
                                                       "bytes": {"fp32": fp_bytes, "int8": int8_bytes}, "latency_ms": {str(k): v for k, v in lat.items()},
                                                       "mask_diff_sim_real": float((m_sim - m_real).abs().mean())})
    if args.save:
        torch.save(real.state_dict(), args.save)


def cmd_evaluate(args, device):
    from .evaluate import evaluate
    cpu = torch.device("cpu")
    module = load_module(args.ckpt, cpu)
    dm = make_datamodule(args, val_crops=1)
    if args.qat:
        sim, _ = load_sim(module, args.qat, cpu)
    else:
        sim = build_sim(module, cfg_from_args(args), dm, cpu, args.calib_crops)
    model = export_int8(sim) if args.real else sim
    q = attach(module, model)
    accel = "cpu" if args.real or args.cpu else "auto"
    if accel == "auto":
        q.to(device)
    evaluate(args.ckpt, args.dataset_path, crop_length=args.crop_length, max_tracks=args.max_tracks, output_json=args.out,
             with_oracle=not args.no_oracle, workers=args.workers, oracle_cache_dir=args.oracle_cache or None, module=q, accelerator=accel)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ptq", "sweep", "sensitivity", "ranges", "qat", "export", "evaluate"])
    ap.add_argument("ckpt")
    ap.add_argument("--dataset-path", default="musdb18/processed")
    ap.add_argument("--split-path", default="musdb18/split.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--calib-crops", type=int, default=64, help="extraits d'entraînement utilisés pour calibrer les plages")
    ap.add_argument("--val-crops", type=int, default=3, help="export : extraits de validation par morceau (3 x 16 = 48)")
    ap.add_argument("--max-batches", type=int, default=None, help="limite le nombre de lots de validation (sensibilité, export)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--cpu", action="store_true", help="force le CPU (la simulation tourne sur GPU par défaut)")
    # réglage de quantification
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--w-per-tensor", action="store_true")
    ap.add_argument("--convt-per-channel", action="store_true")
    ap.add_argument("--w-range", default="minmax", choices=["minmax", "mse"])
    ap.add_argument("--act-range", default="minmax", choices=["minmax", "percentile", "mse"])
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--no-reduce-range", action="store_true")
    ap.add_argument("--fp-input", action="store_true", help="entrée du réseau non quantifiée")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--no-acts", action="store_true")
    # QAT
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", default="a", help="suffixe du fichier .qat_<tag>.pt")
    ap.add_argument("--select", default="last", choices=["last", "best"], help="QAT : garder la dernière époque (défaut) ou la meilleure sur la validation (biaisé vers le haut : choisi sur du bruit)")
    ap.add_argument("--cosine", action="store_true", help="QAT : lr décroissant en cosinus jusqu'à 0 (le lr constant de la baseline oscille de ~0,3 dB d'une époque à l'autre)")
    ap.add_argument("--update-ranges", action="store_true", help="QAT : les plages d'activation suivent une moyenne mobile min/max (Krishnamoorthi) au lieu de rester figées")
    # export / évaluation
    ap.add_argument("--qat", default=None, help="fichier .qat.pt : modèle simulé issu du QAT (sinon PTQ avec le réglage ci-dessus)")
    ap.add_argument("--real", action="store_true", help="evaluate : vrai modèle int8 (CPU) au lieu de la simulation")
    ap.add_argument("--save", default=None, help="export : écrit le state_dict int8")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--crop-length", type=int, default=261632)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("-w", "--workers", type=int, default=None)
    ap.add_argument("--oracle-cache", default="outputs/oracle_cache")
    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    globals()[f"cmd_{args.command}"](args, device)


if __name__ == "__main__":
    main()
