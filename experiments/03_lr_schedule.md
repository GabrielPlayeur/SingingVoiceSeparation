# 03 · Learning rate and scheduler

[← back to README](../README.md)

**Why before anything else.** Every later experiment (losses, compression, size) is compared to the stereo baseline. If the baseline is badly tuned, each conclusion only holds at that setting, and the run-to-run noise is about as large as the effects I expect. So I first looked for a reliable learning rate and scheduler on the baseline alone, then froze them.

**Status.** Done, one seed. Result: the hypothesis is rejected, 1e-3 constant stays.

## Why I suspected the current setting

- `learning_rate = 1e-3` isn't sourced: it's Adam's default, and Jansson et al. 2017 don't give a value ("trained using the ADAM optimizer").
- The papers I read that do give one use lower values, but on other architectures and optimisers, so they don't transfer as-is:

| paper | optimiser | learning rate |
|---|---|---|
| Braun & Tashev (speech, GRU) | AdamW | 1e-4 |
| Wilson et al. (speech, BLSTM) | not stated | 2.1e-4 (best of a Vizier search over 3e-6 to 1e-3) |
| Yamamoto et al. (GAN vocoder) | RAdam | 1e-4 (halved every 200k steps) |
| Choi et al. 2019 | not given in the text | not given |
| **here** | Adam | **1e-3, constant** |

- **What the baseline run showed** (W&B history): best checkpoint at epoch 134 of 150, so early stopping didn't trigger; the run hit the epoch cap. The validation curve was still rising (3.74 dB over epochs 100-110, 3.95 dB over 140-150, about +0.6 dB per 100 epochs): **the model is under-trained**. And it's noisy: the standard deviation of the difference between consecutive epochs is 0.32 dB, and the max (4.14 dB, which is the saved checkpoint) is 0.2 dB above the mean of the last 10 epochs (3.95 ± 0.08 dB). A constant 1e-3 was one possible cause of this noise. BatchNorm and the small validation set are others.

## Hypothesis

A lower learning rate and/or decaying it over the 150-epoch budget gives a stabler curve and a better final model, and therefore less noisy comparisons later.

## Setup

**Scheduler.** Cosine decay **without restarts**: `η_t = η_max/2 · (1 + cos(π t / T))`, `η_min = 0`, updated every step, `T` = total number of steps (150 epochs × 63 steps). It's equation 5 of Loshchilov & Hutter (SGDR, ICLR 2017) with a single cycle. The paper proposes restarts and evaluates on image classification, not audio; I only take the decay. I checked the implementation against the formula (difference 1e-19). No warm-up: the baseline at 1e-3 starts without visible instability. Early stopping is **disabled for the two cosine runs** (it would cut the decay before the end). The constant-LR runs keep the baseline's early stopping (patience 30); only `lr1e4` triggered it, at epoch 129.

**Runs** (stereo baseline, L1 magnitude loss, same data, seed and budget; only the LR and scheduler change):

| config | learning rate | scheduler | where |
|---|---|---|---|
| baseline (run `fgvzjxua`) | 1e-3 | constant | local |
| cosine (`543aaoll`) | 1e-3 | cosine | local |
| lr3e4 | 3e-4 | constant | rented GPU |
| lr3e4_cosine | 3e-4 | cosine | rented GPU |
| lr1e4 | 1e-4 | constant | rented GPU |

3e-4 is a third of the current value and close to Wilson's best (2.1e-4); 1e-4 is Braun's and Yamamoto's value, probably too slow for 150 epochs. A factor of 3 is a usual search step, not a sourced value.

**Decision rule** (written before the runs):
- Choose on **validation only**, never on test, so the test set stays independent for later experiments.
- Criterion: **mean of `val/si_sdr_median` over the last 10 epochs**, not the max, which rewards a lucky epoch. The standard deviation over those 10 epochs gives the noise level; below it, two runs can't be told apart.
- If a setting wins, it goes into `src/configs/model/unet.yaml` and the baseline is retrained with it. Otherwise 1e-3 constant stays, and conclusions are stated as valid for that setting.

## Results

![learning rate study](../figures/lr_schedule.png)

| config | lr | scheduler | epochs | val SI-SDR, mean of last 10 epochs | best epoch (val) |
|---|---|---|---|---|---|
| **baseline** | **1e-3** | **constant** | 150 | **3.95 ± 0.08 dB** | 4.14 dB (ep. 134) |
| cosine | 1e-3 | cosine | 150 | 3.80 ± 0.03 dB | 3.88 dB (ep. 127) |
| lr3e4 | 3e-4 | constant | 150 | 3.73 ± 0.05 dB | 3.84 dB (ep. 138) |
| lr3e4_cosine | 3e-4 | cosine | 150 | 3.47 ± 0.03 dB | 3.56 dB (ep. 110) |
| lr1e4 | 1e-4 | constant | 129 (early stop) | 3.28 ± 0.11 dB | 3.46 dB (ep. 98) |

± = standard deviation over the last 10 epochs (noise within a run). `lr1e4` stopped at epoch 129 (patience 30 after its best epoch 98), so its last 10 epochs are the ones before stopping, and its curve was still slowly rising.

Test (50 tracks, **not used for the choice**, given for the record): SDR median 4.31 dB (lr3e4), 4.36 dB (lr3e4_cosine), 4.05 dB (lr1e4), vs 4.50 dB for the baseline; SI-SDRi median 9.23 / 8.90 / 8.55 dB. On test, `lr3e4_cosine` beats `lr3e4` on SDR (+0.05 dB) but not on SI-SDRi (−0.33 dB), while validation ranks them the other way around. These two can't be separated reliably, which doesn't affect the decision.

## Discussion

- **Hypothesis rejected: nothing beats 1e-3 constant.** The baseline beats each of the four others by 0.15 to 0.68 dB, 2 to 8 times the measured noise (0.08 dB). The ranking on test matches validation for the baseline vs the three server runs. Per the decision rule, **I keep `learning_rate = 1e-3` constant**, and the stereo baseline (SDR 4.50 dB) remains the reference. In the range 1e-4 to 1e-3, the learning rate isn't what limits the model at this budget.
- **For constant LRs, higher is better** (1e-4 < 3e-4 < 1e-3). That fits an **under-trained** baseline (the curve is still rising at epoch 150), not an LR that's too high. A lower LR slows convergence without stabilising enough to make up for it in 150 epochs.
- **Cosine decay doesn't help either:** it loses 0.15 dB at 1e-3 and 0.26 dB at 3e-4 compared to the same constant LR. It does reduce the noise (std 0.03 dB vs 0.05 to 0.08 dB), but run noise wasn't what limited the comparison. My guess (untested): decaying to 0 stops learning while the under-trained model is still improving; with more epochs, cosine might catch up.
- **Both machines give nearly the same result.** `lr3e4_cosine` was trained twice, on the server and locally, with the same seed: 3.47 ± 0.03 dB and 3.52 ± 0.02 dB, a 0.05 dB difference. That's a single pair, so an order of magnitude rather than a measurement of machine variability, but it suggests that moving to the server doesn't shift results by as much as the effects above. *(Later, the baseline itself was trained three times with the same seed, locally and twice on the server: 3.85 to 3.95 dB on this criterion, and within ±0.15 dB paired on test, see [06](06_size_quality.md).)*
- **Limits.** One seed per config: the std I use is within one run, not between seeds. The 0.15 to 0.22 dB gaps (cosine, lr3e4) are above the within-run noise but not proven above seed variability. The decision itself doesn't depend on it (1e-3 was already the value in use), but "cosine does nothing" is a weaker conclusion than "1e-4 and 3e-4 + cosine are worse". The grid stops at 1e-3: given the monotonic trend, a higher LR (3e-3) or more epochs might do better. Not tested; outside the scope of this step.

## Commands

```bash
uv run python -m src.train experiment=stereo_lr3e4          # also stereo_cosine, stereo_lr3e4_cosine, stereo_lr1e4
uv run python -m analysis.lr_schedule --run stereo=fgvzjxua --run cosine=543aaoll \
    --run lr3e4=<run-id> --run lr3e4_cosine=<run-id> --run lr1e4=<run-id>
```

A W&B id is the folder name in `outputs/VoiceSeparation/<id>/`. A server run that wasn't synced is read locally from `outputs/wandb/offline-run-*/run-<id>.wandb`. Output: `figures/lr_schedule.png` and `figures/lr_schedule.json`.

## References

- I. Loshchilov, F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017, [arXiv:1608.03983](https://arxiv.org/abs/1608.03983). Read for equation 5 and the experimental setup.
- LR values of the other papers: Braun & Tashev, Wilson et al. ([05](05_amplitude_compression.md#references)), Yamamoto et al., Choi et al. ([04](04_waveform_loss.md#references)), read in their training sections.

## Re-run at 1000 epochs

At 1000 epochs ([10](10_recipe.md), 3 seeds per cell), a constant rate with a linear decay to 0 over the last 20 % (`wsd`, not the cosine of this page) gains +0.13 dB on validation, below the decision threshold (0.15 dB), and +0.04 / −0.01 dB on test. It does halve the noise of the last epochs (0.04 against 0.09–0.10 dB). The constant 1e-3 stays.
