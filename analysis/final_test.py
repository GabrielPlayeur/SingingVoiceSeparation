"""Test-set results of the re-run (phase 5 of docs/RERUN_PLAN.md): every comparison decided on validation in experiments/10 and 11,
now on the 50 test tracks (official mixture, stereo museval v4 SDR), plus the headline numbers against the 150-epoch baseline and
Open-Unmix.

Nothing here makes a decision: the decisions are on validation (figures/convergence_decide.json, figures/final_decide.json), except
mono, which experiments/11 decides on test because its validation set differs (downmixed crops).

Statistics, per comparison: each configuration is reduced to ONE score per track, the mean over its seeds (3 or 1), then the paired
per-track difference against the reference: median [95 % bootstrap CI over tracks, 10,000 resamples, seed 0], Wilcoxon signed-rank
p, number of tracks where the variant is better. The CI covers track-to-track variability only; the seed spread is reported
separately (median SDR of each seed).

Usage: uv run python -m analysis.final_test     Outputs: figures/final_test.json, figures/final_test.png
"""
import glob
import json
import os

import numpy as np
from scipy import stats

from ._common import *

S = os.path.join(ROOT, "outputs", "server")
RUNS = {  # configuration -> eval.json of each seed (test, official mixture, last checkpoint)
    "const": ["probe-v1/probe_w16_s42", "recipe-v1/r_const_s43", "recipe-v1/r_const_s44"],
    "const_aug": ["probe-v1/probe_w16_aug_s42", "recipe-v1/r_const_aug_s43", "recipe-v1/r_const_aug_s44"],
    "wsd": ["recipe-v1/r_wsd_s42", "recipe-v1/r_wsd_s43", "recipe-v1/r_wsd_s44"],
    "wsd_aug": ["recipe-v1/r_wsd_aug_s42", "recipe-v1/r_wsd_aug_s43", "recipe-v1/r_wsd_aug_s44"],
    "w32": ["probe-v1/probe_w32_aug_s42", "final-v2/f_w32_s43", "final-v2/f_w32_s44"],
    "wsdr": ["final-v2/f_wsdr_s42", "final-v2/f_wsdr_s43", "final-v2/f_wsdr_s44"],
    "sisdr": ["final-v2/f_sisdr_s42"],
    "power_loss": ["final-v2/f_power_loss_s42"],
    "mono": ["final-v2/f_mono_s42"],
    "legacy": ["final-v2/f_legacy_s42"],
    "w32_noaug": ["probe-v1/probe_w32_s42"],
}
EXTERNAL = {  # single evaluations already in outputs/, same protocol (official mixture)
    "baseline_150ep": os.path.join(ROOT, "outputs", "eval_stereo_official.json"),
    "umx": os.path.join(ROOT, "outputs", "umx_niter1.json"),
    "umx_no_wiener": os.path.join(ROOT, "outputs", "umx_niter0.json"),
}
COMPARISONS = [  # (name, variant, reference): variant - reference
    ("augmentation (constant lr)", "const_aug", "const"),
    ("augmentation (wsd)", "wsd_aug", "wsd"),
    ("end decay (no aug)", "wsd", "const"),
    ("end decay (aug)", "wsd_aug", "const_aug"),
    ("w32", "w32", "const_aug"),
    ("wSDR loss", "wsdr", "const_aug"),
    ("SI-SDR loss", "sisdr", "const_aug"),
    ("|X|^0.3 in the loss", "power_loss", "const_aug"),
    ("mono input", "mono", "const_aug"),
    ("head bug (legacy)", "legacy", "const_aug"),
    ("recipe vs 150-epoch baseline", "const_aug", "baseline_150ep"),
    ("final w32 vs 150-epoch baseline", "w32", "baseline_150ep"),
    ("final w32 vs Open-Unmix", "w32", "umx"),
    ("final w32 vs Open-Unmix without Wiener", "w32", "umx_no_wiener"),
]


def load(path):
    with open(path) as f:
        return {r["track"]: r["sdr"] for r in json.load(f)}


def bootstrap_median(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    setup_style()
    per_seed, missing = {}, {}
    for name, runs in RUNS.items():
        found = [r for r in runs if os.path.exists(os.path.join(S, r, "eval.json"))]
        missing[name] = sorted(set(runs) - set(found))
        per_seed[name] = [load(os.path.join(S, r, "eval.json")) for r in found]
    for name, path in EXTERNAL.items():
        per_seed[name] = [load(path)]
        missing[name] = []
    tracks = sorted(set.intersection(*(set(d) for seeds in per_seed.values() for d in seeds)))
    assert len(tracks) == 50, f"{len(tracks)} common tracks"
    score = {k: np.array([np.mean([d[t] for d in seeds]) for t in tracks]) for k, seeds in per_seed.items()}

    out = {"tracks": len(tracks), "configs": {}, "comparisons": {}}
    print("Test SDR, official mixture, median over 50 tracks of the per-track mean over seeds")
    for k, seeds in per_seed.items():
        med, lo, hi = bootstrap_median(score[k])
        seed_meds = [float(np.median([d[t] for t in tracks])) for d in seeds]
        out["configs"][k] = dict(median=med, ci=[lo, hi], seeds=len(seeds), seed_medians=seed_meds, missing=missing[k])
        warn = f"   MISSING {missing[k]}" if missing[k] else ""
        print(f"  {k:15s} {med:5.2f} dB [{lo:.2f}, {hi:.2f}]  seeds {', '.join(f'{m:.2f}' for m in seed_meds)}{warn}")
    print("\nPaired differences (variant - reference), median [95 % CI], Wilcoxon p, wins / 50")
    for name, a, b in COMPARISONS:
        d = score[a] - score[b]
        med, lo, hi = bootstrap_median(d)
        p = float(stats.wilcoxon(d).pvalue)
        wins = int((d > 0).sum())
        out["comparisons"][name] = dict(variant=a, reference=b, median=med, ci=[lo, hi], wilcoxon_p=p, wins=wins)
        print(f"  {name:40s} {med:+.2f} [{lo:+.2f}, {hi:+.2f}]  p={p:.1e}  {wins}/50")

    with open(os.path.join(FIG_DIR, "final_test.json"), "w") as f:
        json.dump(out, f, indent=2)

    names = [n for n, _, _ in COMPARISONS[:10]]
    c = [out["comparisons"][n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    y = np.arange(len(names))[::-1]
    ax.errorbar([x["median"] for x in c], y, xerr=[[x["median"] - x["ci"][0] for x in c], [x["ci"][1] - x["median"] for x in c]],
                fmt="o", color=BLUE, capsize=3)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y, [f"{n} ({out['configs'][x['variant']]['seeds']} vs {out['configs'][x['reference']]['seeds']} seeds)"
                      for n, x in zip(names, c)])
    ax.set_xlabel("paired test SDR difference, median over 50 tracks (dB)")
    ax.set_title("Re-run at 1000 epochs: each change against its reference (official mixture)")
    save(fig, "final_test.png")


if __name__ == "__main__":
    main()
