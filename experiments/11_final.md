# 11 · Re-running the key experiments at 1000 epochs

[← back to README](../README.md)

**In short.** I re-ran the experiments whose conclusion could flip, at 1000 epochs with the recipe of [10](10_recipe.md). **Every decision made on validation held on test.** The final model, w32 with augmentation, scores **5.89 dB**: +0.99 dB over the 150-epoch baseline, under the same 4 GB budget, and 0.27 dB below Open-Unmix at equal reconstruction.

**When this page was written.** Runs, rules and predictions were committed before the runs; the validation results were committed (`8e54341`) before any test evaluation.

## What was re-run

Each run changes one thing against the recipe (w16 stereo, augmentation, constant rate 1e-3, 1000 epochs, last checkpoint):

| run | change | seeds | from |
|---|---|---|---|
| `f_w32` | width 32 | 3 | [06](06_size_quality.md) |
| `f_wsdr` | wSDR loss | 3 | [04](04_waveform_loss.md) |
| `f_sisdr` | SI-SDR loss | 1 | [04](04_waveform_loss.md) |
| `f_power_loss` | `\|X\|^0.3` in the loss | 1 | [05](05_amplitude_compression.md) |
| `f_mono` | mono input | 1 | [02](02_mono_vs_stereo.md) |
| `f_legacy` | old head (the bug) | 1 | [01](01_baseline_vs_oracle.md) |

3 seeds where a decision depended on it (w32 is the candidate final model, wSDR the only promising loss), 1 seed for the others. Not re-run, for time: w4, w8, w64, context length (its conclusion rests on the receptive field, not the budget), `power_in`, `norm`, the MR-STFT loss (it failed by instability) and the learning-rate grid.

**Decision rule** (validation, mean of epochs 991–1000, σ_seed = 0.089 dB from [10](10_recipe.md)): real if the difference exceeds 2·σ_seed·√(2/3) = **0.145 dB** for 3 seeds against 3, or 2·σ_seed·√(4/3) = **0.206 dB** for 1 seed against 3. Below: inconclusive, not "no effect".

## Predictions

1. w32: +0.25 to +0.5 dB with augmentation. The gap should grow with training (H2), not shrink (H1: width only speeds up learning).
2. wSDR: 0 to +0.3 dB, more likely inconclusive.
3. SI-SDR loss: inconclusive.
4. Compressed loss: the gap stays, −0.5 to −1.0 dB.
5. Mono: inconclusive.
6. Head bug: clearly worse, 0.3 to 1.5 dB.

## Validation

| run | Δ vs reference (4.749 dB) | threshold | verdict |
|---|---|---|---|
| w32 (3 seeds) | **+0.458** | 0.145 | real gain |
| wSDR (3 seeds) | **+0.244** | 0.145 | real gain |
| SI-SDR (1) | **−0.689** | 0.206 | real loss |
| compressed loss (1) | **−1.278** | 0.206 | real loss |
| head bug (1) | +0.125 | 0.206 | inconclusive |
| mono (1) | (+0.735) | | not comparable |

**Mono could not be judged on validation.** With `data.channels=1`, the validation crops are built from the mono downmix, which alone raises the oracle by 0.93 dB. My rule hadn't foreseen this (the protocol trap of [02](02_mono_vs_stereo.md) again). I decided mono on test instead, under the stereo protocol, and committed that before computing any test number.

**Against the predictions:**

1. **w32: confirmed, H2.** The gap grows with training: +0.27 dB at epoch 150, +0.45 at 600, +0.46 at 1000. **w32 is the final model** (2.63 GiB peak VRAM).
2. **wSDR: kept**, better than I predicted. It was not trained at width 32, so the final model keeps the L1 loss. Combining the two is a follow-up, not an assumption.
3. **SI-SDR: real loss**, not inconclusive. It is below the reference even on its own training metric. ⚠ No tested explanation; at 150 epochs without augmentation it was +0.07 dB.
4. **Compressed loss: confirmed, larger than predicted.** The gap didn't close, so it wasn't a convergence effect.
5. **Mono: decided on test** (below).
6. **Head bug: rejected.** No measurable cost at this budget (explained below).

## Test

Last checkpoint of every run, 50 tracks, official mixture, stereo museval v4. Each configuration is the per-track mean over its seeds, then the paired difference against the reference. The CI covers tracks only; the seed medians show the seed spread.

![test comparisons](../figures/final_test.png)

| model | test SDR [95 % CI] | seed medians |
|---|---|---|
| 150-epoch baseline | 4.47 [3.54, 5.37] | 4.47 |
| recipe (w16 + augmentation) | 5.45 [3.89, 5.95] | 5.54, 5.38, 5.32 |
| **final: w32 + augmentation** | **5.89 [4.29, 6.30]** | 5.72, 5.82, 5.92 |
| Open-Unmix, no Wiener | 6.06 [5.32, 7.30] | |
| Open-Unmix, Wiener | 6.43 [5.21, 7.24] | |

| change | validation | test, paired [95 % CI] | better on |
|---|---|---|---|
| w32 | +0.46 | **+0.34 [+0.25, +0.40]** | 44/50 |
| wSDR loss | +0.24 | **+0.25 [+0.13, +0.32]** | 39/50 |
| SI-SDR loss | −0.69 | **−0.47 [−0.60, −0.32]** | 5/50 |
| `\|X\|^0.3` in the loss | −1.28 | **−1.00 [−1.21, −0.90]** | 1/50 |
| head bug | +0.13 | −0.01 [−0.13, +0.12] | 23/50 |
| **mono input** | not comparable | **+0.15 [−0.14, +0.28]**, p = 0.41: inconclusive | 29/50 |
| augmentation, constant rate | +0.24 | +0.19 [+0.12, +0.30] | 39/50 |
| augmentation, wsd | +0.27 | +0.30 [+0.16, +0.36] | 44/50 |
| end decay, no augmentation | +0.11 | −0.01 [−0.05, +0.05] | 24/50 |
| end decay, augmentation | +0.14 | +0.04 [−0.00, +0.07] | 31/50 |
| **final w32 vs 150-epoch baseline** | | **+0.99 [+0.88, +1.17]** | 48/50 |
| final w32 vs Open-Unmix | | −0.55 [−0.78, −0.33] | 14/50 |
| final w32 vs Open-Unmix without Wiener | | −0.27 [−0.51, −0.10] | 14/50 |

- **Every validation decision holds on test**, with the same sign. The effects shrink a little (w32 +0.46 → +0.34), as expected when selecting on validation; the two metrics also don't have the same scale.
- **Mono vs stereo: no measurable difference**, as at 150 epochs.
- **The re-run moved the result by about 1 dB** (4.47 → 5.89 dB), better on 48 of 50 tracks, with the same architecture family and the same 4 GB budget. The gap to Open-Unmix went from 1.69 to 0.55 dB, and to 0.27 dB with the same reconstruction.
- The two near-silent tracks (*PR - Happy Daze*, *PR - Oh No*) are still broken for the waveform losses (down to −33 dB): energy-normalised losses never learn to output silence. Medians are not affected.

## Checkpoint analyses

On the 240 validation crops (no test track), except the Wiener filter.

**Head bug** (`analysis.mask_stats`). The 1000-epoch run with the old head learned weights [[−1.06, +1.48], [−1.07, +1.48]]: each channel has one positive weight, so the mask isn't capped (it reaches 1.0). The 0.727 ceiling of the old checkpoint happened because all its head weights were negative. The fix stays, since it removes the failure mode, but I can't claim the bug cost a given number of dB.

**Compressed loss.** The "mask pushed to zero" explanation is rejected: among bins where the voice is weak (ideal ratio between 0.1 and 0.5), the mask is below 0.05 on 6.1 % of them, against 14.7 % for the reference. The codec-weighting explanation of [05](05_amplitude_compression.md) is untested and remains.

**Output level of the waveform losses** (`analysis.output_scale`, 80 crops). What the SNR would gain with the right level: reference 0.14 dB, SI-SDR 0.11 dB, wSDR 0.06 dB. The level doesn't explain why the SI-SDR loss is worse. wSDR has the best-calibrated level, as its scale-sensitive term should give.

**Wiener filter on the final model** ([src/evaluate_wiener.py](../src/evaluate_wiener.py)). Open-Unmix's multichannel Wiener filter (one iteration), applied inside each inference window, with |V| = m·|X| and |A| = (1 − m)·|X|. Seed 42 of w32: **−0.12 dB [−0.20, −0.05]**, better on 13 of 49 tracks. It removes a bit more accompaniment (SIR 9.2 → 10.3 dB) but the SDR goes down, so I don't use it. A plausible reason (⚠ untested): with two sources, the accompaniment model is the exact complement of the vocals model, so the filter has nothing independent to arbitrate between. Open-Unmix has four independent targets. One track (*Punkdisco - Oral Hygiene*) gives a NaN with the filter, even with a 1e-8 floor; it's excluded and I haven't found the cause.

```bash
uv run python -m src.train experiment=recipe model.base_channels=32     # the final model
uv run python -m analysis.final_test
uv run python -m analysis.mask_stats --ckpt reference=<ckpt> --ckpt legacy=<ckpt> --ckpt power_loss=<ckpt>
uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_wiener CKPT -w 0 -o outputs/w32_wiener.json
```
