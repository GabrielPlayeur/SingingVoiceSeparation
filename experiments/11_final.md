# 11 · The 150-epoch experiments again, at budget B and with the phase-2 recipe

[← back to README](../README.md)

**Status.** Prepared before the tag `final-v1`, while phase 2 was still running: hypotheses, predictions and rules below are in git before any phase-3 run starts. The recipe itself (`src/configs/experiment/recipe.yaml`) is filled after the phase-2 decision ([10](10_recipe.md)); everything else on this page does not depend on it.

**Result in one line.** Not available yet.

## Why these runs and not the others

Experiments 02 to 07 were run at 150 epochs, where every model was still improving ([09](09_convergence_probe.md): the baseline gains 0.6 dB on validation between epochs 150 and 1000). Their comparisons mixed "which model is better" with "which model learns faster". This phase re-runs the ones whose conclusion can flip and that matter for the final model, at B = 1000 epochs, against the phase-2 recipe. The 150-epoch results stay in the README as the short-budget column.

Not re-run (decided on 16 Oct for time): w4, w8 and w64 ([06](06_size_quality.md); w64 is over the 4 GB budget anyway), context length ([07](07_context_length.md): its conclusion rests on the 4.42 s receptive field, which does not depend on the budget), `power_in` and `norm` ([05](05_amplitude_compression.md)), the multi-resolution STFT loss ([04](04_waveform_loss.md): it failed by instability, not by lack of training) and the learning-rate grid ([03](03_lr_schedule.md)).

## Runs (group `final` of [deploy/queue.yaml](../deploy/queue.yaml))

One variable changed against `experiment=recipe` (w16 stereo, B = 1000 epochs, no early stopping, last checkpoint):

| run | change | seeds | from |
|---|---|---|---|
| `f_w32` | `model.base_channels=32` | 42, 43, 44 | [06](06_size_quality.md) |
| `f_wsdr` | `model.loss_type=wsdr` | 42, 43, 44 | [04](04_waveform_loss.md) |
| `f_sisdr` | `model.loss_type=sisdr` | 42 | [04](04_waveform_loss.md) |
| `f_power_loss` | `model.loss_power=0.3` | 42 | [05](05_amplitude_compression.md) |
| `f_mono` | `data.channels=1` (and `channel_swap=false`) | 42 | [02](02_mono_vs_stereo.md) |
| `f_legacy` | `model.legacy_head=true` | 42 | [01](01_baseline_vs_oracle.md), the head bug |

Three seeds where a decision depends on it: w32 is the candidate final model, wSDR the only loss that looked promising (+0.31 dB on validation at 150 epochs). The others get one seed: they either settle by a large margin or are reported as inconclusive. Reference: the three seeds of the phase-2 cell chosen as recipe (same code: `git diff recipe-v1 final-v1 -- src/` touches nothing outside `src/configs`).

## Decision rule

- Score of a run: mean of `val/si_sdr_median` over epochs 991 to 1000 (`python -m analysis.convergence at --epoch 1000`). Validation only.
- σ_seed: the pooled seed standard deviation measured in phase 2 (`figures/convergence_decide.json`).
- **3-seed variant against the 3-seed reference**: difference of two means, standard deviation σ_seed·√(2/3); the effect is called real if it exceeds **2·σ_seed·√(2/3)**.
- **1-seed variant against the 3-seed reference**: one run against a mean of three, standard deviation σ_seed·√(1 + 1/3); real if the difference exceeds **2·σ_seed·√(4/3) ≈ 2.31·σ_seed**. Below that: **inconclusive**, not "no effect".
- Verdicts are worded Confirmed / Rejected / Inconclusive against the predictions below.

## Hypotheses and predictions

1. **w32 (H1 against H2 of [06](06_size_quality.md)).** H1: the +0.22 dB of w32 at 150 epochs came from faster convergence, and the gap shrinks at B. H2: with augmentation the larger model has data to use its capacity, and the gap grows. The probe (one seed) supports H2 with augmentation (+0.07 dB at epoch 150, +0.49 dB at epoch 1000) and H1 without it (+0.35 → +0.26 dB). **Prediction: w32 − w16 between +0.25 and +0.5 dB if the recipe has augmentation, between +0.1 and +0.3 dB without it.** H1 and H2 are read from the gap at epoch 150 against epoch 1000 of the same runs (validation is logged every epoch). Peak VRAM of w32 on the GTX 1650: 2.63 GiB ([06](06_size_quality.md)), inside the budget.
2. **wSDR.** At 150 epochs: +0.31 dB on validation, +0.12 dB [−0.10, +0.36] on test. Near convergence the mismatch between an amplitude loss and a waveform metric may matter more. **Prediction: between 0 and +0.3 dB; more likely inconclusive than kept.**
3. **SI-SDR loss.** At 150 epochs: +0.11 dB test SDR, no SI-SDRi gain. **Prediction: inconclusive.** ⚠ Bias of the rule: this model optimises exactly the validation metric, so its validation gain is an upper bound; if it passes the threshold, the decision is only confirmed if the test SDR of phase 5 agrees.
4. **Compression in the loss (`|X|^0.3`).** At 150 epochs: −0.82 dB test, −1.1 dB validation. Two explanations ([05](05_amplitude_compression.md)): slower convergence (then the gap shrinks at B), or the weight the compressed loss gives to the bins above 16 kHz that the AAC codec emptied (12 times more than the linear loss), which does not depend on the budget. **Prediction: the gap stays, between −0.5 and −1.0 dB** (I expect the codec weighting to dominate). A gap below 0.3 dB in absolute value would reject this and point to convergence.
5. **Mono.** At 150 epochs: +0.04 dB [−0.18, +0.17] for stereo. **Prediction: inconclusive** (below 2.31·σ_seed).
6. **Head bug.** The mask is capped at sigmoid(b) (0.727 in the old checkpoint, i.e. at least −2.8 dB on bins dominated by the voice) and constant on the bins where the head ReLU is off. SI-SDR is scale-invariant, but this cap is not a uniform gain: it distorts the estimate where the voice dominates. **Prediction: clearly worse, by 0.3 to 1.5 dB** (⚠ wide range: no clean measurement exists, the only pre-fix test score is not comparable, see [01](01_baseline_vs_oracle.md)).

**If a prediction fails.** Each run says something either way: a w32 gain that shrinks (H1) means width was mostly a speed-up; a `power_loss` gap that closes means the 150-epoch result was a convergence artefact; a head bug that costs nothing means the network learned around it.

## Results

*To be filled from the phase-3 runs; nothing above will be edited.*
