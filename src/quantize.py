"""int8 quantization of a checkpoint: PTQ, ablations, per-layer sensitivity, QAT, real export, evaluation (experiments/08).

Everything runs locally (4 GB GPU for the simulation and QAT, CPU for the real int8). Subcommands:

    python -m src.quantize ptq CKPT            # one simulated PTQ setting: validation SI-SDR fp32 vs quantized
    python -m src.quantize sweep CKPT          # grid of PTQ ablations (bit width, range, granularity...) -> JSON
    python -m src.quantize sensitivity CKPT    # one quantizer at a time (Nagel § 3.7) -> JSON
    python -m src.quantize ranges CKPT         # observed range vs quantiles of each activation -> JSON
    python -m src.quantize qat CKPT            # fine-tuning with fake quantization -> .qat.pt checkpoint
    python -m src.quantize export CKPT         # real int8 (CPU): size, latency, agreement with the simulation
    python -m src.quantize evaluate CKPT       # museval SDR on the test tracks with the simulated or real model
    python -m src.quantize profile CKPT        # time of each (de)convolution, fp32 vs real int8 (CPU) -> JSON

The fast metrics (median validation SI-SDR) are used for the ablations; the final decision is read on the test SDR.
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
# Model and data
# ----------------------------------------------------------------------------------------------------------------------

def load_module(ckpt, device):
    module = VoiceSeparationModule.load_from_checkpoint(ckpt, map_location="cpu").eval()
    return module.to(device)


def make_datamodule(args, stage="fit", val_crops=None):
    """`val_crops`: validation crops per track (15 by default = those of training, ~1 GB in memory; 1 when they are not
    needed, e.g. for the evaluation on the test set)."""
    cfg = OmegaConf.load(CONFIG_DATA)
    cfg.dataset_path, cfg.split_path = args.dataset_path, args.split_path
    if val_crops is not None:
        cfg.val_crops_per_track = val_crops
    cfg.channels, cfg.num_workers, cfg.batch_size = 2, args.num_workers, args.batch_size
    dm = MUSDB18DataModule(**cfg)
    dm.setup(stage)
    return dm


def val_loader(dm):
    """Validation crops already in memory: no reader processes (one per worker would copy ~0.5 GB each, this PC has 8 GB of RAM)."""
    from torch.utils.data import DataLoader
    return DataLoader(dm.val_dataset, batch_size=dm.hparams.batch_size, shuffle=False, num_workers=0)


def train_loader(dm, num_workers):
    """Training crops. "spawn" process (not "fork"): a forked worker inherits the ~0.5 GB of validation crops in memory
    and copies them (reference counts), which overflows this PC's 8 GB of RAM as soon as another computation is running."""
    from torch.utils.data import DataLoader
    return DataLoader(dm.train_dataset, batch_size=dm.hparams.batch_size, shuffle=True, num_workers=num_workers,
                      persistent_workers=num_workers > 0, multiprocessing_context="spawn" if num_workers > 0 else None)


def network_inputs(module, batch, device):
    """U-Net input (transformed amplitude) + what is needed to rebuild the waveform, like `validation_step`."""
    batch = {k: v.to(device) for k, v in batch.items()}
    batch = module.on_after_batch_transfer(batch, 0)
    return batch


@torch.no_grad()
def val_sisdr(module, loader, device, max_batches=None):
    """Validation SI-SDR (median, mean) on fixed crops: the same quantity as the training `val/si_sdr_median`."""
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
    """`n_crops` random TRAINING crops (fixed seed) -> network inputs. No validation or test track
    (Nagel § 3.5: a small unlabelled calibration set is enough; ~100 batches in Krishnamoorthi § 3.1.2)."""
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
    """Replaces the module's U-Net with the simulated version: `module(...)`, `_to_wave`, `evaluate` can then be used as is."""
    q = VoiceSeparationModule(**module.hparams)  # same hyper-parameters, same STFT window; the original module stays intact
    q.model = sim
    try:
        device = next(sim.parameters()).device
    except StopIteration:   # real int8 model: packed weights, no parameters; it only runs on CPU
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
    print(f"{cfg}\nfp32  val SI-SDR median {ref['median']:.3f} dB (n={ref['n']})\nquant val SI-SDR median {q['median']:.3f} dB  "
          f"(difference {q['median'] - ref['median']:+.3f} dB)")


SWEEP = [  # (name, QuantConfig overrides): one variable at a time around the base W8A8 minmax setting
    ("fp32 (control: quantizers off)", dict(quantize_weights=False, quantize_acts=False)),
    ("weights only W8 (per channel)", dict(quantize_acts=False)),
    ("weights only W8 (per tensor everywhere)", dict(quantize_acts=False, w_per_channel=False, convt_per_channel=False)),
    ("weights only W8 (ConvT per channel)", dict(quantize_acts=False, convt_per_channel=True)),
    ("weights only W8 (MSE range)", dict(quantize_acts=False, w_range="mse")),
    ("weights only W6", dict(quantize_acts=False, w_bits=6)),
    ("weights only W4", dict(quantize_acts=False, w_bits=4)),
    ("activations only A8 (minmax)", dict(quantize_weights=False)),
    ("activations only A8 (percentile 99.99)", dict(quantize_weights=False, act_range="percentile")),
    ("activations only A8 (percentile 99.9)", dict(quantize_weights=False, act_range="percentile", percentile=99.9)),
    ("activations only A8 (mse)", dict(quantize_weights=False, act_range="mse")),
    ("activations only A8, full 8 bits (no reduce_range)", dict(quantize_weights=False, reduce_range=False)),
    ("activations only A8, input in float", dict(quantize_weights=False, quantize_input=False)),
    ("activations only A8 (percentile 99.99), full 8 bits", dict(quantize_weights=False, act_range="percentile", reduce_range=False)),
    ("W8A8 minmax (default setting)", dict()),
    ("W8A8 minmax, full 8 bits", dict(reduce_range=False)),
    ("W8A8 percentile 99.99, full 8 bits", dict(act_range="percentile", reduce_range=False)),
    ("W8A8 percentile 99.99", dict(act_range="percentile")),
    ("W8A8 mse (weights and activations)", dict(act_range="mse", w_range="mse")),
    ("W8A8 mse, input in float", dict(act_range="mse", w_range="mse", quantize_input=False)),
    ("W8A8 percentile 99.99, input in float", dict(act_range="percentile", quantize_input=False)),
]


def cmd_sweep(args, device):
    module = load_module(args.ckpt, device)
    dm = make_datamodule(args)
    loader = val_loader(dm)
    batches = calibration_batches(module, dm, device, args.calib_crops)
    ref = val_sisdr(module, loader, device)
    print(f"fp32: val SI-SDR median {ref['median']:.3f} dB (n={ref['n']})")
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
    """Nagel § 3.7 "per-layer analysis": ONE single quantizer is quantized (the others stay in float) and the drop is measured."""
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
    """Observed range of each activation (min, max) against its 0.01 % and 99.99 % quantiles: explains why the min-max range is
    bad (a quantization step set by a few extreme values)."""
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
    print(f"{'quantizer':14s} {'min':>9s} {'p0.01':>9s} {'p99.99':>9s} {'max':>9s} {'max/p99.99':>11s} {'8-bit step (minmax)':>20s} {'step (p99.99)':>13s}")
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
    print(f"fp32 {ref['median']:.3f} dB | PTQ {ptq['median']:.3f} dB ({ptq['median'] - ref['median']:+.3f}) | QAT: {args.epochs} epochs, lr {args.lr}{' cosine' if args.cosine else ''}", flush=True)
    params = [p for p in sim.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    steps_per_epoch = len(range(0, dm.hparams.epoch_size, dm.hparams.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=0.0) if args.cosine else None
    # quant: activation ranges frozen at calibration (percentile...); qat: min/max moving average; off: control without quantization
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
        print(f"epoch {epoch:2d}  loss {tot / n:.4f}  val SI-SDR {v['median']:.3f} dB ({v['median'] - ref['median']:+.3f} vs fp32)  {time.time() - t0:.0f} s", flush=True)
        if args.select == "best" and v["median"] > best or args.select == "last" and epoch == args.epochs:
            best, best_state = v["median"], {k: x.detach().cpu().clone() for k, x in sim.state_dict().items()}
    if best_state is None:
        print("QAT did not beat the starting PTQ: keeping the PTQ state (no checkpoint written).")
        return
    sim.load_state_dict(best_state)
    sim.eval(); sim.set_mode(act="quant" if cfg.quantize_acts else "off", weights="quant" if cfg.quantize_weights else "off"); sim.finalize_weights()
    out = args.out or args.ckpt.replace(".ckpt", "") + f".qat_{args.tag}.pt"
    torch.save({"state_dict": sim.state_dict(), "config": vars(cfg), "ckpt": args.ckpt, "history": history, "fp32_val": ref, "ptq_val": ptq,
                "epochs": args.epochs, "lr": args.lr, "cosine": args.cosine}, out)
    print(f"state kept ({args.select}): {best:.3f} dB -> {out}")


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
    print(f"val SI-SDR median: fp32 {r_fp['median']:.3f} | simulation {r_sim['median']:.3f} | real int8 {r_real['median']:.3f} dB (n={r_fp['n']})")
    x = next(iter(val_loader(dm)))
    b = network_inputs(module, {k: v[:1] for k, v in x.items()}, cpu)
    inp = module_input(module, b["mix_amp"])
    with torch.no_grad():
        m_fp, m_sim, m_real = module.model(inp), sim(inp), real(inp)
    print(f"mask: mean |sim - real| {float((m_sim - m_real).abs().mean()):.5f} (max {float((m_sim - m_real).abs().max()):.3f}) ; "
          f"mean |fp32 - real| {float((m_fp - m_real).abs().mean()):.5f}")
    fp_bytes, int8_bytes = state_dict_bytes(module.model), state_dict_bytes(real)
    print(f"size: fp32 {fp_bytes / 1e6:.2f} MB (BatchNorm included) | int8 {int8_bytes / 1e6:.2f} MB ({fp_bytes / int8_bytes:.2f}x)")
    lat = {}
    for threads in args.threads:
        lat[threads] = {"fp32": latency_ms(module.model, inp, threads), "int8": latency_ms(real, inp, threads)}
        f, i = lat[threads]["fp32"][0], lat[threads]["int8"][0]
        print(f"latency 1 crop 5.93 s, {threads} thread(s): fp32 {f:.0f} ms | int8 {i:.0f} ms ({f / i:.2f}x)")
    save_json(args.out or "figures/quant_export.json", {"ckpt": args.ckpt, "qat": args.qat, "val": {"fp32": r_fp, "sim": r_sim, "real": r_real},
                                                       "bytes": {"fp32": fp_bytes, "int8": int8_bytes}, "latency_ms": {str(k): v for k, v in lat.items()},
                                                       "mask_diff_sim_real": float((m_sim - m_real).abs().mean())})
    if args.save:
        torch.save(real.state_dict(), args.save)


def cmd_profile(args, device):
    """CPU time of each Conv2d / ConvTranspose2d, fp32 against real int8, measured layer by layer on their real input
    (captured by hooks during a forward pass). The input values do not affect the time: random crop
    of the shape of a training crop (1, channels, n_fft/2 + 1, frames)."""
    cpu = torch.device("cpu")
    module = load_module(args.ckpt, cpu)
    if args.qat:
        sim, _ = load_sim(module, args.qat, cpu)
    else:
        sim = build_sim(module, cfg_from_args(args), make_datamodule(args), cpu, args.calib_crops)
    hp = module.hparams
    x = torch.rand(1, hp.in_channels, hp.n_fft // 2 + 1, args.crop_length // hp.hop_length + 1)
    models = {"fp32": module.model.eval(), "int8": export_int8(sim).eval()}

    def layer_times(model, threads):
        caught = []
        hooks = [m.register_forward_hook(lambda m, i, o: caught.append((m, i[0]))) for m in model.modules()
                 if "Conv" in type(m).__name__ and not list(m.children())]
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        rows = []
        with torch.no_grad():
            for m, inp in caught:
                rows.append({"layer": type(m).__name__, "ms": latency_ms(m, inp, threads, repeats=5)[0]})
        return rows

    out = {"ckpt": args.ckpt, "qat": args.qat, "input_shape": list(x.shape), "threads": {}}
    for threads in args.threads:
        res = {name: layer_times(m, threads) for name, m in models.items()}
        summary = {}
        for name, rows in res.items():
            tr = [r["ms"] for r in rows if "Transpose" in r["layer"]]
            cv = [r["ms"] for r in rows if "Transpose" not in r["layer"]]
            summary[name] = {"conv_transpose_ms": tr, "conv_ms": cv}
            print(f"{threads} thread(s), {name} : {len(tr)} ConvTranspose2d {sum(tr):.0f} ms {[round(t) for t in tr]} | "
                  f"{len(cv)} Conv2d {sum(cv):.0f} ms")
        a, b = summary["fp32"]["conv_transpose_ms"], summary["int8"]["conv_transpose_ms"]
        print(f"  int8 / fp32 per deconvolution: {[round(j / i, 1) for i, j in zip(a, b)]} (total {sum(b) / sum(a):.1f}x); "
              f"convolutions: {sum(summary['int8']['conv_ms']) / sum(summary['fp32']['conv_ms']):.2f}x")
        out["threads"][str(threads)] = summary
    save_json(args.out or "figures/quant_profile.json", out)


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
    ap.add_argument("command", choices=["ptq", "sweep", "sensitivity", "ranges", "qat", "export", "evaluate", "profile"])
    ap.add_argument("ckpt")
    ap.add_argument("--dataset-path", default="musdb18/processed")
    ap.add_argument("--split-path", default="musdb18/split.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--calib-crops", type=int, default=64, help="training crops used to calibrate the ranges")
    ap.add_argument("--val-crops", type=int, default=3, help="export: validation crops per track (3 x 16 = 48)")
    ap.add_argument("--max-batches", type=int, default=None, help="limits the number of validation batches (sensitivity, export)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--cpu", action="store_true", help="forces the CPU (the simulation runs on GPU by default)")
    # quantization setting
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--w-per-tensor", action="store_true")
    ap.add_argument("--convt-per-channel", action="store_true")
    ap.add_argument("--w-range", default="minmax", choices=["minmax", "mse"])
    ap.add_argument("--act-range", default="minmax", choices=["minmax", "percentile", "mse"])
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--no-reduce-range", action="store_true")
    ap.add_argument("--fp-input", action="store_true", help="network input not quantized")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--no-acts", action="store_true")
    # QAT
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", default="a", help="suffix of the .qat_<tag>.pt file")
    ap.add_argument("--select", default="last", choices=["last", "best"], help="QAT: keep the last epoch (default) or the best on validation (biased upwards: chosen on noise)")
    ap.add_argument("--cosine", action="store_true", help="QAT: lr decaying as a cosine to 0 (the baseline's constant lr oscillates by ~0.3 dB from one epoch to the next)")
    ap.add_argument("--update-ranges", action="store_true", help="QAT: the activation ranges follow a min/max moving average (Krishnamoorthi) instead of staying frozen")
    # export / evaluation
    ap.add_argument("--qat", default=None, help=".qat.pt file: simulated model from QAT (otherwise PTQ with the setting above)")
    ap.add_argument("--real", action="store_true", help="evaluate: real int8 model (CPU) instead of the simulation")
    ap.add_argument("--save", default=None, help="export: writes the int8 state_dict")
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
