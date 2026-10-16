# 04 · Waveform losses

[← back to README](../README.md)

**Status.** Done, one seed (tag `loss-v1`, 4 runs trained in parallel on the rented GPU). The hypothesis, prediction, design choices and protocol below were written before training; results start at [Results](#results).

**Result in one line.** None of the three losses reaches the predicted +0.3 dB. SI-SDR and wSDR are about +0.1 dB SDR, the same size as the difference between two trainings of the same baseline. Multi-resolution STFT is clearly worse (−0.5 dB). I keep the L1 magnitude loss.

**Baseline.** The stereo baseline with lr 1e-3 constant ([03](03_lr_schedule.md) kept it). It was retrained on the server alongside these runs (`baseline` in the `loss` group of [deploy/queue.yaml](../deploy/queue.yaml)) so the comparison is on one machine. That rerun scores 4.47 dB SDR, vs 4.50 dB for the local baseline used as the reference elsewhere.

## Hypothesis

The baseline loss (L1 on magnitudes) isn't what I measure. Evaluation compares *waveforms*, obtained after the inverse STFT with the mixture's phase. The loss only looks at magnitudes before reconstruction, so it ignores the effect of the mixture phase and of the iSTFT (a modified STFT isn't necessarily consistent anymore). Training directly on the reconstructed waveform (`_to_wave` in `src/module.py` is already differentiable) should bring the optimisation closer to the SDR / SI-SDR I evaluate.

**Prediction** (an estimate, not sourced): +0.3 to 1 dB SDR. **If rejected:** the mismatch between the L1 magnitude loss and SDR isn't the limiting factor (and the mixture phase is, which the oracle already quantified), or what works on speech doesn't transfer to music.

## What the papers say

| | Le Roux et al. | Choi et al. | Yamamoto et al. |
|---|---|---|---|
| contribution | defines **SI-SDR** and shows the flaws of BSS Eval SDR | **weighted-SDR (wSDR)**, a waveform loss after iSTFT | **multi-resolution STFT** loss (spectral convergence + log magnitude) |
| task | metric definition; examples on speech (CHiME2), comparison on wsj0-2mix | speech enhancement (VoiceBank-DEMAND, 16 kHz), complex mask bounded by tanh | neural speech vocoder (TTS, 24 kHz) |
| what is shown | SDR and SI-SDR differ by about 0.5 dB (0.3 to 0.6) on wsj0-2mix; BSS Eval SDR can stay high while part of the spectrum is removed | "almost every result shows that optimizing with wSDR loss gives the best result" vs spectrogram MSE and waveform MSE, over 3 masks × 3 network sizes; but spectrogram MSE often gives a better PESQ (DCU-10, DCU-16) | the multi-resolution loss gives a better MOS than the same loss at a single resolution; it "prevents the generator from being overfit to a fixed STFT representation" |

**What they don't show, which limits my conclusions:**
- None of them is about **music separation**. Choi (noisy speech, 16 kHz, complex mask), Yamamoto (speech synthesis, judged by listening) and Le Roux (a metric) are about other things. I'm testing whether the idea transfers to a singing voice in a dense mix.
- Le Roux et al. **don't propose** training with SI-SDR: the paper defines the metric and cites other work that uses it as a loss ("outperforming least-squares on some tasks"). Using it as a loss is my extrapolation.
- Choi's "phase-aware" argument comes from a **complex** mask that corrects the phase. Here the mask is real in [0, 1] and the phase is the mixture's, fixed: the loss can only act on the magnitude. The gain is bounded by that ceiling (oracle 9.09 dB vs 4.50 dB for the model: the gap is mostly the model, not the magnitude ceiling).
- Yamamoto doesn't define resolutions for 44.1 kHz or for source separation: see my choices below.

## Runs

One variable changes vs the stereo baseline. Code in `src/utils/losses.py`, hooked in `VoiceSeparationModule._loss`, configs in `src/configs/experiment/`.

| config | loss (ŷ = iSTFT(mask · \|X\|, mixture phase)) | source | command |
|---|---|---|---|
| baseline | L1(\|X̂\|, \|S\|) on magnitudes | Jansson et al. 2017 | `experiment=stereo` |
| sisdr | −SI-SDR(ŷ, y) | Le Roux et al. eq. 4-5 | `uv run python -m src.train experiment=stereo_sisdr` |
| wsdr | −[α·cos(y, ŷ) + (1−α)·cos(z, ẑ)], z = x − y, ẑ = x − ŷ, α = ‖y‖²/(‖y‖²+‖z‖²) | Choi et al. eq. 7 and 9 | `uv run python -m src.train experiment=stereo_wsdr` |
| mrstft | mean over 3 resolutions of ‖\|S(y)\|−\|S(ŷ)\|‖_F/‖\|S(y)\|‖_F + mean\|log\|S(y)\|−log\|S(ŷ)\|\| | Yamamoto et al. eq. 3-6, table 1 | `uv run python -m src.train experiment=stereo_mrstft` |

Design choices and what they change:
- **SI-SDR** is the same function as `val/si_sdr_*` (`utils.si_sdr`, averaged over channels). So this model optimises exactly the validation metric. That's on purpose, but it means the decision has to be made on **test museval SDR**, not on validation SI-SDR, where this model has an unfair advantage.
- **wSDR** is computed per example and per channel, then averaged; `y` = vocals, `z = x − y` = accompaniment (the mixture really is vocals + accompaniment, `dataset.py`). Epsilon 1e-8 in the denominators (the paper adds one without giving the value). Bounded in [−1, 1] (checked: −1.000 when ŷ = y). Unlike a plain cosine, the noise term ẑ = x − ŷ makes it **scale-sensitive** (checked: −0.947 for ŷ = 0.5·y or 1.5·y), which is what the paper wants.
- **mrstft:** the paper's resolutions (windows of 25 / 50 / 10 ms, hops of 5 / 10 / ~2 ms) **converted to samples at 44.1 kHz keeping the durations**: (FFT, window, hop) = (2048, 1102, 220), (4096, 2205, 441), (1024, 441, 92). That's my adaptation; the paper only covers 24 kHz. Weight 1 between the two terms, as in the paper. A floor of 1e-7 on \|STFT\|², because the derivative of \|z\| doesn't exist at 0 (my implementation choice). No waveform L1 term: it would need a weight nothing justifies (on one batch with an untrained model, waveform L1 is 0.033 vs 2.0 for the MR-STFT loss, so adding it unweighted would amount to ignoring it).
- `val/loss` is only **comparable between models with the same loss**, so early stopping and checkpointing use `val/si_sdr_median`, shared by all experiments.
- **Cost** (measured after warm-up, mean of 10 steps, batch 8 stereo, GTX 1650, excluding data loading and validation): baseline 0.29 s/step and 1.37 GiB peak VRAM; sisdr and wsdr 0.32 s/step and 1.60 GiB; mrstft 0.70 s/step (2.4× the baseline) and 2.38 GiB thanks to recomputing the resolutions during backprop (3.42 GiB without). All fit in 4 GB.

**Checks done.** SI-SDR and wSDR recomputed by hand give the same values (7.386 dB; −0.919); `mrstft(y, y) = 0`; gradients are finite (including a target with exact digital silence and ŷ = 0); the gradient with recomputation is identical to the one without; the baseline (`loss_type=amp_l1`) is exactly the previous L1; the old stereo checkpoint still loads; a 4-step training + validation + save + reload works for all 3 losses.

## Is the output level a risk?

SI-SDR doesn't penalise a gain error, while museval SDR (and SNR) does: a −SI-SDR loss gives no signal about the level of the estimated voice. Measured on the baseline (80 validation crops, `uv run python -m analysis.output_scale`):

![output level](../figures/output_scale.png)

- Median SI-SDR 4.26 dB, median SNR 5.32 dB. SI-SDR isn't bounded by SNR (Le Roux, § 2.2), so the gap has no fixed sign.
- Median optimal gain α −0.77 dB (p10 −3.92, p90 +0.33): the estimated voice is slightly **too loud**.
- Cost of the level, SNR(α·ŝ) − SNR(ŝ): 0.13 dB median (p90 0.83). The baseline's level is already about right; it's the reference to compare to after training.

Prediction to test: for `sisdr` the level cost goes up (the mask bounded in [0, 1] limits the drift, but nothing corrects it); `wsdr` and `mrstft` (spectral convergence relative to ‖S(y)‖) are sensitive to it. If the cost stays around 0.1 dB, the risk is ruled out. **Not tested yet**: it needs the checkpoints, which I didn't bring back from the server (see [What's left](#whats-left)).

## Results

`uv run python -m analysis.loss_study` (reads `outputs/server/loss-v1/<run>_s42/`: `eval.json` and the offline W&B history). Figure and numbers in `figures/loss_study.png` / `.json`.

![waveform losses](../figures/loss_study.png)

**How to read the middle and right panels.** Each dot is one test track: the score of the variant minus the score of the server baseline on the same track. The two grey columns are two other trainings of the *same* baseline (same config, same seed 42): `rerun_local` was trained on my GTX 1650, `rerun_w16` is the w16 point of the size curve ([06](06_size_quality.md)), trained later on the server with identical code. They show how much two trainings of the same model differ, so they're the yardstick for everything else: their paired SDR differences stay within about ±0.15 dB.

**Validation** (`val/si_sdr_median`, mean of the last 10 epochs ± std, the criterion from [03](03_lr_schedule.md)):

| loss | val SI-SDR, last 10 epochs | best epoch | epoch-to-epoch noise |
|---|---|---|---|
| baseline (L1 magnitude) | 3.85 ± 0.11 dB | 4.04 dB (ep. 145) | 0.34 dB |
| sisdr | 3.92 ± 0.08 dB | 4.05 dB (ep. 144) | 0.28 dB |
| **wsdr** | **4.16 ± 0.09 dB** | 4.32 dB (ep. 128) | 0.24 dB |
| mrstft | 3.21 ± 0.13 dB | 3.42 dB (ep. 145) | 1.03 dB |

**Test** (50 tracks, paired against the server baseline; 95% bootstrap CI of the median difference, Wilcoxon p):

| | test SDR median | Δ SDR vs baseline | better on | Δ SI-SDRi vs baseline | Δ SDR without the 2 *PR* tracks |
|---|---|---|---|---|---|
| baseline (server) | 4.47 dB | | | | |
| *baseline rerun (local)* | *4.50 dB* | *+0.06 [−0.07, +0.15], p = 0.22* | *30/50* | *−0.19 [−0.24, −0.00], p = 0.17* | *+0.06 [−0.05, +0.15]* |
| *baseline rerun (w16 of [06](06_size_quality.md))* | *4.78 dB* | *+0.01 [−0.06, +0.15], p = 0.43* | *27/50* | *−0.09 [−0.23, +0.00], p = 0.07* | *+0.01 [−0.06, +0.15]* |
| sisdr | 4.71 dB | +0.11 [+0.01, +0.31], p = 0.02 | 32/50 | −0.01 [−0.16, +0.17], p = 0.75 | +0.15 [+0.04, +0.33], p = 0.003 |
| wsdr | 4.63 dB | +0.12 [−0.10, +0.36], p = 0.10 | 29/50 | +0.06 [−0.21, +0.25], p = 0.86 | +0.18 [−0.08, +0.39], p = 0.02 |
| mrstft | 3.88 dB | **−0.51 [−0.71, −0.41]**, p = 4e-8 | 10/50 | **−0.80 [−0.91, −0.63]**, p = 4e-8 | −0.51 [−0.71, −0.41] |

Training time on the server (4 runs sharing one A800): 24.4 min for the baseline, 25.7 sisdr, 25.8 wsdr, 33.1 mrstft.

## Discussion

- **The hypothesis is rejected in the form I predicted.** I expected +0.3 to 1 dB SDR. The best losses give about +0.1 dB in median, and mrstft loses 0.5 dB.
- **SI-SDR and wSDR: a small gain I can't separate from noise.** Their median SDR gains (+0.11 and +0.12 dB) are about the size of the gaps between trainings of the same baseline (up to +0.06 dB SDR with CIs reaching +0.15, and up to 0.19 dB SI-SDRi in the other direction). Placed on the size curve ([06](06_size_quality.md)), they give +0.13 and +0.14 dB against w16, with CIs that include 0, while simply doubling the width gives +0.22 dB. For sisdr the CI excludes 0, but that CI only covers track-to-track variability, not training-to-training. On SI-SDRi neither does anything (−0.01 and +0.06 dB). It's also a bit surprising that the model trained *on* SI-SDR doesn't improve test SI-SDRi at all. One difference: training sees 6 s crops with an audible voice, while the test scores whole tracks, including quiet passages.
- **wSDR has the best validation curve of any w16 model** (4.16 dB, +0.31 dB over the baseline, about 3 times the within-run noise and 3 times the largest gap between the three baseline trainings on validation, 3.85 to 3.95 dB). The test doesn't clearly confirm it. It's the one candidate worth testing again with several seeds.
- **All three waveform losses collapse on *PR - Happy Daze*** (−15 to −16 dB SDR), the test track where the voice is almost inaudible ([00](00_dataset.md)). That's what pulls the **mean** differences of sisdr and wsdr below zero (−0.13 and −0.22 dB) while their medians are positive. This track is unstable anyway: the other two trainings of the same baseline already differ from this one by 6.9 and 9.2 dB on it. A possible reason (not tested): SI-SDR, wSDR and spectral convergence are all normalised by the energy of the target, and training crops without voice are filtered out (`silence_threshold`), so these losses never teach the model to output near-silence when the voice is quiet. The output-level check above would test this.
- **mrstft is clearly worse and unstable.** −0.51 dB SDR and −0.80 dB SI-SDRi, better on only 10 tracks out of 50, far beyond the rerun noise. Its validation curve has sudden collapses: at epoch 70 it drops from 2.63 to −1.41 dB, and at epoch 84 from 2.96 to −1.03 dB, while the training loss barely moves (1.45 → 1.48 and 1.39 → 1.46). So the training itself isn't diverging; it's the model in evaluation mode that jumps. Two untested explanations:
  - the log-magnitude term weights every bin equally whatever its energy. With 22.8% of bins almost empty (AAC cutoff above ~16 kHz, see [05](05_amplitude_compression.md)), part of the loss is spent matching codec noise around the 1e-7 floor;
  - BatchNorm: evaluation uses running statistics, and a loss that pushes the network into a different regime may make them less reliable.
  The resolutions are also my own adaptation from 24 kHz to 44.1 kHz. So this result says "this MR-STFT setup doesn't work here", not "MR-STFT doesn't work".

## Decision

**I keep the L1 magnitude loss as the baseline loss.** No loss beats the run-to-run noise on test SDR, which was the decision metric. mrstft is ruled out in this form. wSDR stays as a candidate: if the 3-seed runs show the baseline's spread is well under 0.1 dB, I'll rerun wSDR with 3 seeds before deciding again.

## What's left

- Bring back the sisdr and wsdr checkpoints (`deploy/fetch_results.sh loss-v1 --ckpt sisdr_s42 --ckpt wsdr_s42`) and run the output-level analysis, to test the prediction above and the *Happy Daze* explanation.
- Seed variability of the baseline (`seeds` group), to know what a +0.1 dB difference is worth.

## Commands

```bash
# server (tag loss-v1, commit 5e4050e)
.venv/bin/python deploy/run_queue.py --tag loss-v1 --group loss --parallel 4
# PC
deploy/fetch_results.sh loss-v1 --wait
uv run python -m analysis.loss_study --dir outputs/server/loss-v1 \
    --rerun rerun_local=outputs/eval_stereo.json --rerun rerun_w16=outputs/server/size-v1/w16_s42/eval.json
uv run python -m analysis.output_scale --ckpt baseline=<baseline ckpt> --ckpt sisdr=<sisdr ckpt>   # not run yet
```

## References

- J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, *SDR – half-baked or well done?*, [arXiv:1811.02508](https://arxiv.org/abs/1811.02508), 2018. Read in full. The arXiv page doesn't give the venue (ICASSP 2019 from memory, not checked).
- H.-S. Choi, J.-H. Kim, J. Huh, A. Kim, J.-W. Ha, K. Lee, *Phase-aware Speech Enhancement with Deep Complex U-Net*, ICLR 2019, [arXiv:1903.03107](https://arxiv.org/abs/1903.03107). Read in full.
- R. Yamamoto, E. Song, J.-M. Kim, *Parallel WaveGAN: a fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram*, ICASSP 2020, [arXiv:1910.11480](https://arxiv.org/abs/1910.11480). Read the multi-resolution loss section and table 1.
