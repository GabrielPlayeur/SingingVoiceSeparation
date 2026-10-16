"""Oracle variants: how much of the ceiling comes from the mixture phase, from the [0, 1] bound of the mask, from the mask itself?

The reference oracle (`src.evaluate`, experiment 01) applies |V|/|X| bounded to [0, 1] to the mixture, with the mixture phase.
One factor is changed at a time (see `src.models.oracle.Oracle`):
  - irm_mix  : bounded |V|/|X|, mixture phase (reference, must give 9.09 dB again);
  - irm_true : same amplitude, phase of the true vocals → cost of the mixture phase, bounded mask unchanged;
  - none_mix : exact amplitude |V| (unbounded mask), mixture phase → cost of the ≤ 1 bound;
  - psm_mix  : phase-sensitive mask bounded to [0, 1] (Erdogan et al. 2015), mixture phase → best real mask
               possible for the model's sigmoid output (in the least-squares sense, not in SDR);
  - none_true: |V| with the phase of V, control (exact STFT/iSTFT reconstruction): only the SNR, no museval.
The scoring protocol is that of `src.evaluate.score_track` (stereo museval v4, 1 s windows, accompaniment =
mixture − estimated vocals; SI-SDRi).

Usage:
    python -m analysis.oracle_phase compute [-w 2]       # computes what is missing in outputs/oracle_variants.json (~2.5 GB/process)
    python -m analysis.oracle_phase plot [--model baseline=outputs/eval_stereo.json --model w32=outputs/server/size-v1/w32_s42/eval.json]
Outputs: outputs/oracle_variants.json (per-track scores), figures/oracle_phase.png, figures/oracle_phase.json
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._common import *

CACHE = os.path.join(ROOT, "outputs", "oracle_variants.json")
N_FFT, HOP, SR = 2048, 512, 44100  # those of the model (src/configs/model/unet.yaml)
VARIANTS = {  # name -> (mask, phase) of src.models.oracle.Oracle
    "irm_mix": ("irm", "mixture"),
    "irm_true": ("irm", "true"),
    "none_mix": ("none", "mixture"),
    "psm_mix": ("psm", "mixture"),
}
LABELS = {
    "irm_mix": "bounded IRM, mixture phase\n(reference oracle)",
    "psm_mix": "phase-sensitive mask,\nmixture phase",
    "none_mix": "exact amplitude |V|,\nmixture phase",
    "irm_true": "bounded IRM,\nphase of the true vocals",
}


def score(task):
    """One (track, variant) in a pool process: same metrics as src.evaluate.score_track."""
    import torch
    import librosa
    from src.data.dataset import MUSDBFullTrackDataset
    from src.evaluate import median_sdr, si_sdri
    from src.models.oracle import Oracle

    sample = MUSDBFullTrackDataset(track_dirs=[task["track_dir"]], sr=SR, channels=2)[0]
    mixture_t, vocals_t = sample["mixture"], sample["vocals"]
    mixture, vocals = mixture_t.numpy(), vocals_t.numpy()
    if task["variant"] == "none_true":
        # control + statistics of the ideal mask, per channel
        est = Oracle(N_FFT, HOP, mask="none", phase="true").evaluate(mixture, vocals)
        snr = 10 * np.log10(np.sum(vocals ** 2) / np.sum((vocals - est) ** 2))
        over, over_energy, cos_w, energy = 0, 0.0, 0.0, 0.0
        n_bins = 0
        for m, v in zip(mixture, vocals):
            X = librosa.stft(m, n_fft=N_FFT, hop_length=HOP, pad_mode="reflect")
            V = librosa.stft(v, n_fft=N_FFT, hop_length=HOP, pad_mode="reflect")
            p = np.abs(V) ** 2
            above = np.abs(V) > np.abs(X)
            over += int(above.sum())
            n_bins += above.size
            over_energy += float(p[above].sum())
            cos_w += float((p * np.cos(np.angle(V) - np.angle(X))).sum())
            energy += float(p.sum())
            del X, V, p, above
        return {"track": task["track"], "variant": "none_true", "snr": float(snr),
                "frac_bins_mask_gt1": over / n_bins, "frac_vocal_energy_mask_gt1": over_energy / energy,
                "cos_phase_diff_vocal_weighted": cos_w / energy}
    mask, phase = VARIANTS[task["variant"]]
    est = Oracle(N_FFT, HOP, mask=mask, phase=phase).evaluate(mixture, vocals)
    return {"track": task["track"], "variant": task["variant"],
            "sdr": median_sdr(vocals, mixture - vocals, est, mixture - est, SR),
            "si_sdri": si_sdri(torch.from_numpy(est), mixture_t, vocals_t)}


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {"meta": {"n_fft": N_FFT, "hop_length": HOP, "dataset": os.path.relpath(PROCESSED, ROOT)}, "tracks": {}}


def compute(workers):
    data = load_cache()
    tasks = [{"track": name_of(d), "track_dir": d, "variant": v}
             for d in track_dirs()["test"] for v in (*VARIANTS, "none_true")
             if v not in data["tracks"].get(name_of(d), {})]
    print(f"{len(tasks)} computations to do ({workers} processes)")

    def collect(r):
        name, v = r.pop("track"), r.pop("variant")
        data["tracks"].setdefault(name, {})[v] = r
        tmp = CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, CACHE)  # can resume after an interruption
        print(f"  {name[:40]:40s} {v:9s} " + "  ".join(f"{k}={x:.3f}" for k, x in r.items()), flush=True)

    if workers == 0:
        for t in tasks:
            collect(score(t))
        return
    os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for fut in cf.as_completed([pool.submit(score, t) for t in tasks]):
            collect(fut.result())


def paired(a, b):
    """Median [95 % CI] of a − b, paired tracks, + number of tracks where a > b + Wilcoxon p."""
    d = np.asarray(a) - np.asarray(b)
    m, lo, hi = bootstrap_ci(d)
    return {"median": m, "ci": [lo, hi], "better_on": int((d > 0).sum()), "n": len(d),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}


def plot(models):
    setup_style()
    data = load_cache()["tracks"]
    tracks = sorted(t for t, r in data.items() if all(v in r for v in (*VARIANTS, "none_true")))
    print(f"{len(tracks)} complete tracks")
    sdr = {v: np.array([data[t][v]["sdr"] for t in tracks]) for v in VARIANTS}
    sis = {v: np.array([data[t][v]["si_sdri"] for t in tracks]) for v in VARIANTS}
    for name, path in models:
        with open(path) as f:
            rows = {r["track"]: r for r in json.load(f)}
        sdr[name] = np.array([rows[t]["sdr"] for t in tracks])
        sis[name] = np.array([rows[t]["si_sdri"] for t in tracks])
        ref = np.array([rows[t]["sdr_oracle"] for t in tracks])
        print(f"control {name}: max |sdr_oracle(eval.json) − irm_mix| = {np.max(np.abs(ref - sdr['irm_mix'])):.4f} dB")

    out = {"tracks": len(tracks), "levels": {}, "paired_vs_irm_mix": {}, "paired_vs_irm_true": {}, "mask_stats": {}}
    print("\nMedian over the tracks [95 % bootstrap CI]")
    for v in (*[m for m, _ in models], *VARIANTS):
        m, lo, hi = bootstrap_ci(sdr[v])
        ms, los, his = bootstrap_ci(sis[v])
        out["levels"][v] = {"sdr": [m, lo, hi], "si_sdri": [ms, los, his]}
        print(f"  {v:9s} SDR {m:6.2f} [{lo:.2f}, {hi:.2f}]   SI-SDRi {ms:6.2f} [{los:.2f}, {his:.2f}]")
    print("\nPaired SDR differences, relative to irm_mix")
    for v in [*[m for m, _ in models], "psm_mix", "none_mix", "irm_true"]:
        out["paired_vs_irm_mix"][v] = p = paired(sdr[v], sdr["irm_mix"])
        print(f"  {v:9s} {p['median']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]  better on {p['better_on']}/{p['n']}  p={p['wilcoxon_p']:.1e}")
    for name, _ in models:
        out["paired_vs_irm_true"][name] = p = paired(sdr[name], sdr["irm_true"])
        print(f"  {name} − irm_true : {p['median']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]")
    for k in ("snr", "frac_bins_mask_gt1", "frac_vocal_energy_mask_gt1", "cos_phase_diff_vocal_weighted"):
        x = np.array([data[t]["none_true"][k] for t in tracks])
        out["mask_stats"][k] = {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}
        print(f"  {k:30s} median {np.median(x):.4f}  [min {x.min():.4f}, max {x.max():.4f}]")
    rho, _ = stats.spearmanr(sdr["irm_mix"], sdr["irm_true"] - sdr["irm_mix"])
    out["spearman_irm_mix_vs_phase_gain"] = float(rho)
    print(f"  Spearman(reference oracle, gain from the true phase) = {rho:.2f}")

    # figure: (1) median levels, (2) paired differences per track
    order = [*[m for m, _ in models], "irm_mix", "psm_mix", "none_mix", "irm_true"]
    labels = {**{m: f"model {m}" for m, _ in models}, **LABELS}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    for yi, v in zip(y, order):
        m, lo, hi = out["levels"][v]["sdr"]
        color = ORANGE if v in dict(models) else BLUE
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([m], [yi], color=color, s=60, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{m:.2f}", (m, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [labels[v] for v in order], fontsize=8.5)
    ax.set_xlabel("median SDR over 50 tracks (dB), 95 % bootstrap CI")
    ax.set_title("Models and oracle variants")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    diffs = [v for v in order if v != "irm_mix"]
    rng = np.random.default_rng(0)
    y = np.arange(len(diffs))[::-1]
    for yi, v in zip(y, diffs):
        d = sdr[v] - sdr["irm_mix"]
        color = ORANGE if v in dict(models) else BLUE
        ax.scatter(d, yi + rng.uniform(-0.18, 0.18, len(d)), s=14, color=color, alpha=0.45, edgecolor="none")
        p = out["paired_vs_irm_mix"][v]
        ax.plot(p["ci"], [yi + 0.32] * 2, color=INK, linewidth=2, solid_capstyle="round")
        ax.scatter([p["median"]], [yi + 0.32], color=INK, s=36, zorder=3)
        ax.annotate(f"{p['median']:+.2f} dB", (p["median"], yi + 0.32), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax.axvline(0, color=INK_2, linestyle="--", linewidth=1)
    ax.set_yticks(y, [labels[v].replace("\n", " ") if len(labels[v]) < 40 else labels[v] for v in diffs], fontsize=8.5)
    ax.set_xlabel("paired gap with the reference oracle (SDR dB), one point per track")
    ax.set_title("Where does the ceiling go?")
    ax.grid(axis="y", visible=False)
    save(fig, "oracle_phase.png")
    with open(os.path.join(FIG_DIR, "oracle_phase.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("-> figures/oracle_phase.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("-w", "--workers", type=int, default=2, help="processes (0 = sequential), ~2.5 GB of RAM each")
    p = sub.add_parser("plot")
    p.add_argument("--model", action="append", default=None, metavar="NOM=eval.json")
    args = ap.parse_args()
    if args.cmd == "compute":
        compute(args.workers)
    else:
        models = args.model or ["baseline=outputs/eval_stereo.json", "w32=outputs/server/size-v1/w32_s42/eval.json"]
        plot([tuple(m.split("=", 1)) for m in models])


if __name__ == "__main__":
    main()
