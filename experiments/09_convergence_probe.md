# 09 · Convergence probe (fixing the training budget B)

[← back to README](../README.md)

**Status.** Running on the rented GPU (tag `probe-v1`, 4 runs in parallel). **This page was written while the probe was running, not before it.** The tag `probe-v1` points at commit `4e2ba94` (16 Oct 2026, 14:17 +08), which contains no prediction page. What is fixed in git before any curve was read: the rule that gives B, as constants of [`analysis/convergence.py`](../analysis/convergence.py) (commit `c9ccd33`, 16 Oct 2026, 15:16 +08), and the experiment configs (`probe_*`, commit `52d58fa`). I had not opened any probe curve when I wrote the predictions below. I can't prove that from git, so read them accordingly.

**Result in one line.** Not available yet.

## Why

All the runs of experiments 02 to 07 stop at 150 epochs (9,450 steps) while validation is still rising, so their comparisons measure how fast a model converges as much as where it ends. Early stopping makes it worse: the epoch-to-epoch noise of the validation score (0.32 dB for w16, 1.11 dB for w32, [03](03_lr_schedule.md) and [06](06_size_quality.md)) is larger than the progress over its patience window (about 0.18 dB over 30 epochs), so a run stops on a lucky peak. Before re-running anything I need one budget B, the same for every run, chosen from where a long run actually flattens.

## Hypothesis

1. **Without augmentation**, validation flattens and then goes down: with 84 training tracks the model starts to overfit. The gap between validation loss and training loss grows.
2. **With augmentation** (stem remix, gain, channel swap, [`MUSDBTrainDataset`](../src/data/dataset.py)), the curve keeps rising for longer.
3. If both hold, augmentation could not have helped at 150 epochs, because the overfitting regime had not been reached. That would change how the 150-epoch results must be read.

**If rejected** (the plain runs keep rising to 1,000 epochs, or the augmented ones flatten as early as the plain ones): the 150-epoch comparisons were limited by something else than overfitting, and B is set by the rule below anyway. A result that contradicts (1) is as informative as one that confirms it.

Planning assumption, not a prediction: the cost estimates of the next phases use B ≈ 600 epochs.

## Setup

| run | config | epochs |
|---|---|---|
| P1 | `probe_w16`: w16, no augmentation, LR 1e-3 constant | 1,000 |
| P2 | `probe_w16_aug`: w16, augmentation | 1,000 |
| P3 | `probe_w32`: w32, no augmentation | 1,000 |
| P4 | `probe_w32_aug`: w32, augmentation | 1,000 |

Seed 42, early stopping off, validation on the 16 validation tracks, last checkpoint kept. P1 and P2 save a weights-only checkpoint every 100 epochs. w32 is in the probe because B must hold for the largest model compared under 4 GB. One seed only: B is a planning value, not a result to be compared between configurations.

## Decision protocol (frozen in `analysis/convergence.py`)

- Metric: `val/si_sdr_median`, smoothed by a trailing mean over 20 epochs.
- Gain: smoothed value 100 epochs later minus smoothed value now. B is the first epoch at which this gain falls below **0.05 dB per 100 epochs**, rounded up to the next multiple of 100 (so it lands on a periodic checkpoint). 0.05 dB is a third of the ±0.15 dB between reruns of the same baseline ([06](06_size_quality.md)): below it, the remaining gain can't be told from noise.
- B of the study = the largest B of P2 and P4 (augmentation is the recipe I expect to keep, and w32 is the largest model). Curve still rising at epoch 1,000: B = 1,000.
- **First crossing, not "stays below until the end".** I chose this before seeing probe data, from the noise measured on the 150-epoch baseline (0.18 dB standard deviation of the epoch-to-epoch differences). With that noise, the gain of a 20-epoch mean fluctuates by about 0.04 dB, close to the 0.05 dB threshold: in a simulation (4(1 − e^(−e/100)) plus noise) the "stays below" variant returned B of 800 to 900 whatever the curve, the first crossing returned 400 (500 without noise). The first crossing can be early by up to a hundred epochs; the script prints the largest gain after the crossing as a check.
- Overfitting check: `python -m analysis.convergence plot` draws validation, training and validation loss, and their gap. The training loss of the augmented runs is measured on augmented data, so gaps are compared within a run, not between augmented and plain runs.

## Results

*To be filled when the curves are fetched. The numbers will be pasted from `figures/convergence_budget.json`; nothing above will be edited.*

## What's left

- Phase 2 reads the seed-42 constant-LR cells from these runs at epoch B (`analysis.convergence at --epoch B`), so only seeds 43 and 44 are trained for them.
- The probe's last checkpoints stay on the server for the final test (phase 5).
