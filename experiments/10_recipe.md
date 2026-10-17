# 10 · Training recipe: augmentation × end-of-training decay

[← back to README](../README.md)

**In short.** With 3 seeds per configuration at 1000 epochs, **data augmentation is kept** (+0.25 dB on validation). A learning-rate decay at the end of training is not kept (+0.13 dB, below the threshold), although it halves the noise of the last epochs. Recipe for everything after: constant rate 1e-3 + augmentation. The noise between seeds is now measured: **σ_seed = 0.089 dB**.

**When this page was written.** During the runs, after the tag. Fixed in git beforehand: the decision rule (`analysis.convergence decide`), B = 1000 and the 10 configs. No curve had been opened when I wrote the predictions.

## Question

Every later run is compared to one baseline. At 1000 epochs, should it use augmentation, and should the learning rate decay at the end?

## Design: 2 × 2, 3 seeds per cell

| | constant rate 1e-3 | constant, then linear decay to 0 over the last 20 % (`wsd`) |
|---|---|---|
| **no augmentation** | `const` | `wsd` |
| **augmentation** (remix + gain + L/R swap) | `const_aug` | `wsd_aug` |

- w16 stereo, L1 loss, 1000 epochs, no early stopping, last checkpoint.
- Seed 42 of the two constant cells is the probe run itself ([09](09_convergence_probe.md)): with a constant rate, a run doesn't depend on its total length.
- Why `wsd` and not the cosine of [03](03_lr_schedule.md): cosine lowers the rate from the start and cost 0.15 dB on a model still learning. `wsd` keeps the full rate for 80 % of the run (Hägele et al. 2024, arXiv:2405.18392; ⚠ the 20 % is my choice, not checked against the paper).
- Augmentation values are Open-Unmix's defaults (gain uniform in [0.25, 1.25] per source, L/R swap with p = 0.5), checked in its source. Remix is on 2 sources only: the processed data has vocals and accompaniment, not the 4 stems.

## Predictions

1. Augmentation: +0.15 to +0.35 dB, kept.
2. End decay: 0 to +0.2 dB, and the noise of the last epochs at least halved.
3. (Weak) the decay helps more without augmentation.

## Decision rule

- Score of a run: mean of `val/si_sdr_median` over epochs 991 to 1000. Validation only.
- σ_seed: standard deviation between seeds, pooled over the 4 cells (8 degrees of freedom).
- A factor is kept if its main effect exceeds **2 · σ_seed · √(2/3)**, twice the standard deviation of a difference between two 3-seed means.

## Results

| cell | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| const | 4.504 | 4.427 | 4.600 | 4.511 |
| **const_aug** | 4.743 | 4.632 | 4.871 | **4.749** |
| wsd | 4.661 | 4.692 | 4.512 | 4.622 |
| wsd_aug | 4.872 | 4.878 | 4.919 | 4.890 |

σ_seed = **0.089 dB**, threshold **0.145 dB**.

| factor | effect | decision |
|---|---|---|
| augmentation | **+0.253 dB** (+0.238 at constant rate, +0.268 with wsd) | **kept** |
| end decay | +0.126 dB (+0.111 without aug, +0.141 with) | not kept |
| interaction | +0.030 dB (noise: 0.206) | none |

The decay does reduce the noise of the last 10 epochs: 0.036–0.043 dB against 0.094–0.099 dB.

## Against the predictions

1. **Augmentation: confirmed**, inside the predicted range, same size in both rows.
2. **End decay: inconclusive.** Positive in both rows but below the threshold. Its noise prediction is confirmed (2.2 to 2.8 times lower). I followed the rule and didn't adopt it: changing a rule a second time after seeing the numbers, without a failed pre-written check like the one in [09](09_convergence_probe.md), would not be defensible. The cost: final checkpoints carry about 0.1 dB of epoch-to-epoch noise.
3. **Interaction: rejected, weakly.** The decay helps slightly *more* with augmentation, and the interaction is far below its noise.

On test ([11](11_final.md)), augmentation gives +0.19 / +0.30 dB and the decay −0.01 / +0.04 dB: both decisions hold.

**What it changes.** σ_seed = 0.089 dB replaces the "±0.15 dB rerun noise" of the 150-epoch pages, which only measured GPU non-determinism with one seed. Two single runs must now differ by more than about 0.25 dB before the difference means anything.

```bash
uv run python -m analysis.convergence decide --budget 1000 \
  --cell const=outputs/server/probe-v1/probe_w16_s42,outputs/server/recipe-v1/r_const_s43,outputs/server/recipe-v1/r_const_s44 \
  --cell const_aug=outputs/server/probe-v1/probe_w16_aug_s42,outputs/server/recipe-v1/r_const_aug_s43,outputs/server/recipe-v1/r_const_aug_s44 \
  --cell wsd=outputs/server/recipe-v1/r_wsd_s42,outputs/server/recipe-v1/r_wsd_s43,outputs/server/recipe-v1/r_wsd_s44 \
  --cell wsd_aug=outputs/server/recipe-v1/r_wsd_aug_s42,outputs/server/recipe-v1/r_wsd_aug_s43,outputs/server/recipe-v1/r_wsd_aug_s44
```
