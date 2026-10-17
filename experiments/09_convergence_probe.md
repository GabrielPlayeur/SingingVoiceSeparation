# 09 · Convergence probe: choosing the training budget

[← back to README](../README.md)

**In short.** Experiments 02 to 07 stopped at 150 epochs while every model was still improving. Four runs of 1000 epochs show that the baseline gains another 0.6 dB, and that **data augmentation, which costs 0.4 to 0.7 dB at 150 epochs, gains 0.24 to 0.47 dB at 1000**. My pre-written rule gave a budget of 600 epochs, but its own check failed, so I took **B = 1000** and wrote down why.

**When this page was written.** During the probe, not before it. What was fixed in git before any curve was read: the rule for B (constants of [`analysis/convergence.py`](../analysis/convergence.py), commit `c9ccd33`) and the run configs (`52d58fa`). I hadn't opened a probe curve when I wrote the predictions, but git can't prove it.

## Why

At 150 epochs, comparisons measure how fast a model learns as much as how good it ends up. Early stopping makes it worse: the validation score moves by 0.32 dB from one epoch to the next (1.11 dB for w32), more than the progress over the 30-epoch patience (about 0.18 dB), so runs stop on a lucky peak. I needed one budget B for every run, taken from where long runs actually flatten.

## Hypotheses

1. **Without augmentation**, validation flattens and then goes down (overfitting on 84 tracks).
2. **With augmentation** (stem remix, gain, L/R swap), it keeps rising for longer.
3. If both hold, augmentation could not have helped at 150 epochs.

## Setup

| run | width | augmentation |
|---|---|---|
| w16 | 16 | no |
| w16_aug | 16 | yes |
| w32 | 32 | no |
| w32_aug | 32 | yes |

1000 epochs each, seed 42, no early stopping, last checkpoint kept. w32 is included because B must also hold for the largest model under 4 GB.

**Rule for B (frozen in code):** smooth `val/si_sdr_median` over 20 epochs; B is the first epoch where the gain over the next 100 epochs falls below 0.05 dB, rounded up to a multiple of 100; take the largest B of the two augmented runs. Check: print the largest gain *after* that crossing.

## Results

![convergence probe](../figures/convergence.png)

| validation SI-SDR (mean of 10 epochs) | ep. 150 | ep. 600 | ep. 1000 |
|---|---|---|---|
| w16 | 3.90 | 4.40 | 4.50 |
| w16_aug | 3.47 | 4.42 | **4.74** |
| w32 | 4.25 | 4.76 | 4.76 |
| w32_aug | 3.54 | 4.89 | **5.23** |

**What the rule gives:** B = 600 (w16_aug crosses the threshold at epoch 567, w32_aug at 389).

**Why I took 1000 instead.** The check fails: after crossing the threshold, the augmented curves rise again by up to 0.18 and 0.29 dB per 100 epochs, 4 to 6 times the tolerance. Between epochs 600 and 1000 they still gain +0.32 and +0.34 dB, while the plain runs are flat. The crossing was a noise dip. At B = 600, w16_aug and w16 are equal (4.42 vs 4.40); at 1000, augmentation is ahead by 0.24 dB. Phase 2 would have concluded "augmentation does nothing" for exactly the reason this re-run exists. So I took B = 1000, the length of the probe.

This is a deviation from my own rule, decided after seeing the curves. What makes it defensible: the check that failed was written beforehand, B applies to every variant, no test number had been computed, and the deviation goes towards training longer. Its weak point: the argument uses the augmentation gap, which is what phase 2 then measures.

## Hypotheses, checked

1. **Half confirmed.** Without augmentation, validation flattens (w32 around epoch 500, w16 around 700) but doesn't go down by epoch 1000.
2. **Confirmed.** With augmentation, +0.32 / +0.34 dB between epochs 600 and 1000, against +0.10 / 0.00 without.
3. **Confirmed, and stronger than I thought.** Augmentation *costs* 0.43 dB (w16) and 0.71 dB (w32) at epoch 150, and *gains* 0.24 and 0.47 dB at epoch 1000. The ranking flips with the budget. The 150-epoch pages could not see this.

```bash
uv run python -m analysis.convergence budget --run w16_aug=outputs/server/probe-v1/probe_w16_aug_s42 --run w32_aug=outputs/server/probe-v1/probe_w32_aug_s42
uv run python -m analysis.convergence plot --budget 1000 --run w16=... --run w16_aug=... --run w32=... --run w32_aug=...
```
