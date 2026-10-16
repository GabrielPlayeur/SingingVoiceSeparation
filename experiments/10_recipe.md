# 10 · Training recipe: augmentation × end-of-training learning-rate decay

[← back to README](../README.md)

**Status.** Running on the rented GPU (tag `recipe-v1`, 10 runs, `--parallel 5`). **This page was written while phase 2 was running, after the tag**: the tag `recipe-v1` (commit `d6bafbb`, 16 Oct 2026, 18:41 +08) contains the queue entries but no prediction page. What was fixed in git before the runs: the decision rule, as code and constants of [`analysis/convergence.py`](../analysis/convergence.py) (`decide`, commit `c9ccd33`), the budget B = 1000 ([09](09_convergence_probe.md)) and the 10 run configs. No phase-2 curve had been opened when the predictions below were written.

**Result in one line.** Not available yet.

## Question

Every experiment of phase 3 is compared to one baseline. Which training recipe should that baseline use, at the budget B = 1000 epochs: with or without data augmentation, with a constant learning rate or with a decay at the end?

## Design: 2 × 2 × 3 seeds

| | constant lr 1e-3 | constant, then linear decay to 0 over the last 20 % (`wsd`) |
|---|---|---|
| **no augmentation** | `const`: probe_w16_s42 + r_const_s43, s44 | `wsd`: r_wsd_s42, s43, s44 |
| **augmentation** (remix + gain + L/R swap) | `const_aug`: probe_w16_aug_s42 + r_const_aug_s43, s44 | `wsd_aug`: r_wsd_aug_s42, s43, s44 |

- w16 stereo, L1 amplitude loss, 1000 epochs (63,000 steps), no early stopping, last checkpoint.
- Seed 42 of the two constant-lr cells is the probe run itself: with a constant learning rate the run does not depend on its total length, and B = 1000 is the probe's length. `git diff probe-v1 recipe-v1 -- src/` only touches `evaluate.py` and `evaluate_umx.py` (evaluation code), so the training path is the same.
- Why `wsd` and not the cosine of [03](03_lr_schedule.md): the cosine lowers the rate from the first step and cost 0.15 dB at 150 epochs, on a model that was still learning. `wsd` keeps the full rate for 80 % of the budget and only cools down at the end (Hägele et al. 2024, arXiv:2405.18392; the 20 % is a choice of this project, ⚠ not checked against the paper's recommended fraction). Its purpose is a final checkpoint that is not one random point of a noisy curve: at constant lr the score moves by 0.3 dB from one epoch to the next ([03](03_lr_schedule.md)).
- Augmentation values: gain uniform in [0.25, 1.25] per source and L/R swap with p = 0.5 per source, the defaults of Open-Unmix (`openunmix/data.py`, checked in the source); remix on 2 sources only (the processed data holds vocals and accompaniment, not the 4 stems).

## Hypotheses and predictions

1. **Augmentation helps at this budget.** The probe gives +0.24 dB on validation for w16 (one seed). Prediction: main effect between +0.15 and +0.35 dB, kept.
2. **The end decay helps a little and reduces the noise of the final checkpoint.** Prediction: main effect between 0 and +0.2 dB, and a standard deviation over the last 10 epochs at least halved (0.03 dB against 0.08 dB at 150 epochs in [03](03_lr_schedule.md)). Whether it passes the threshold is uncertain.
3. **Interaction (weak expectation, reported, not used for the decision): the decay helps more without augmentation.** The plain model is on its plateau by epoch 1000 ([09](09_convergence_probe.md)), where a cool-down only removes parameter noise, its usual gain. The augmented model is still rising slowly (+0.04 dB between epochs 800 and 1000 for w16_aug), so cooling down also cuts some learning.

**If rejected.** If augmentation does not pass the threshold, the +0.24 dB of the probe was within seed noise, and phase 3 runs without it. If the decay does not pass, the final-checkpoint noise is not worth a change of recipe and the constant lr stays.

## Decision rule (frozen in `analysis.convergence decide`)

- Score of a run: mean of `val/si_sdr_median` over epochs 991 to 1000. Validation only, never test.
- σ_seed pooled over the 4 cells (8 degrees of freedom).
- A factor is kept if its main effect (mean of its two simple effects, 6 runs against 6) exceeds **2 · σ_seed · √(2/3)**, twice the standard deviation of a difference of two 3-seed means. For a 6-against-6 contrast the true standard deviation is σ_seed · √(1/3), so this threshold is conservative by √2.
- If the two simple effects of a factor have opposite signs, the script flags it and the decision is taken per cell, not on the main effect.
- The recipe printed by the script becomes `src/configs/experiment/recipe.yaml`, the base of phase 3 ([11](11_final.md)).

```bash
uv run python -m analysis.convergence decide --budget 1000 \
  --cell const=outputs/server/probe-v1/probe_w16_s42,outputs/server/recipe-v1/r_const_s43,outputs/server/recipe-v1/r_const_s44 \
  --cell const_aug=outputs/server/probe-v1/probe_w16_aug_s42,outputs/server/recipe-v1/r_const_aug_s43,outputs/server/recipe-v1/r_const_aug_s44 \
  --cell wsd=outputs/server/recipe-v1/r_wsd_s42,outputs/server/recipe-v1/r_wsd_s43,outputs/server/recipe-v1/r_wsd_s44 \
  --cell wsd_aug=outputs/server/recipe-v1/r_wsd_aug_s42,outputs/server/recipe-v1/r_wsd_aug_s43,outputs/server/recipe-v1/r_wsd_aug_s44
```

## Known limits, stated before the results

- Three seeds per cell: σ_seed is estimated with 8 degrees of freedom, so the threshold itself is uncertain (by roughly ±25 %, ⚠ order of magnitude).
- The decision is on validation SI-SDR (16 tracks, 240 crops), not on test SDR; the two have disagreed before ([03](03_lr_schedule.md)).
- Augmentation is tested as one block (remix + gain + swap): a gain would not say which of the three helps.
- The training loss of the augmented cells is measured on augmented data: it cannot be compared with the plain cells.

## Results

*To be filled from `figures/convergence_decide.json`; nothing above will be edited.*
