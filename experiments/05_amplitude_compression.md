# 05 · Amplitude compression

[← back to README](../README.md)

**In short.** Speech papers compress the magnitude (`|X|^0.3`) to give more weight to quiet bins. Here, compressing it **in the loss** costs about 0.8 dB at 150 epochs and **1.00 dB at 1000 epochs**, the opposite of what I predicted. Compressing or normalising only the input changes nothing measurable. I keep the plain baseline.

## Hypothesis

The L1 loss on raw magnitudes is dominated by loud bins (low frequencies). A power law `|X|^c`, with c = 0.3 as in Braun & Tashev and Wilson et al., should give more weight to quiet bins such as the voice's high frequencies.

**Prediction:** a modest gain (under 0.5 dB) for compression in the loss.

Both papers are about speech enhancement, not music. Their evidence is about the **loss**; compressing the input is my extrapolation from Wilson et al.

## What the network receives

![input amplitudes](../figures/input_amplitude_levels.png)

- |X| spans **58 dB** between its median (0.059) and its 99.9th percentile (47).
- 22.8 % of bins are almost empty (|X| < 1e-3): the AAC codec cut everything above ~16 kHz.
- Track loudness varies by 10 dB between the 10th and 90th percentile.

The baseline is sensitive to volume: turning the input up or down by 12 dB costs about 1 dB SI-SDR, and 2.6 dB at −18 dB. A power law doesn't fix that; only per-crop normalisation (`norm`) does.

![gain sensitivity](../figures/input_gain_sensitivity.png)

## Setup

One variable changes against the stereo baseline. The mask is still applied to the linear magnitude.

| config | network input | loss |
|---|---|---|
| baseline | \|X\| | L1 on \|X\| |
| power_loss | \|X\| | L1 on \|X\|^0.3 |
| power_in | \|X\|^0.3 | L1 on \|X\| |
| power_both | \|X\|^0.3 | L1 on \|X\|^0.3 |
| norm | \|X\| / its mean over the crop | L1 on \|X\| |

## Results (150 epochs, one seed)

![amplitude compression](../figures/power_study.png)

| | val SI-SDR Δ | test Δ SDR, paired [95 % CI] | better on |
|---|---|---|---|
| *same baseline, retrained* | | *+0.06 [−0.07, +0.15]* | *30/50* |
| power_loss | −1.11 | **−0.82 [−1.01, −0.64]** | 5/50 |
| power_in | −0.18 | +0.07 [−0.09, +0.22] | 29/50 |
| power_both | −1.06 | **−0.98 [−1.11, −0.76]** | 5/50 |
| norm | −0.05 | −0.01 [−0.19, +0.19] | 24/50 |

- **Compression in the loss hurts clearly**, 5 to 10 times the rerun noise. I checked the code path and didn't find a bug.
- **Compressing the input alone, or normalising it, is neutral on SDR.** `norm` was meant for volume robustness, which the test (tracks at their original level) can't show.
- All four variants make validation noisier from one epoch to the next (0.60 to 2.10 dB, against 0.34 dB).

## Why does the compressed loss hurt?

| explanation | status |
|---|---|
| Slower convergence | **rejected** at 1000 epochs: the gap grows (−0.82 → −1.00 dB) |
| `x^0.3` is steep near 0, so it pushes the mask to 0 where the voice is weak | **rejected**: among the weak-voice bins, the mask is below 0.05 on 6.1 % of them, against 14.7 % for the reference (`analysis.mask_stats`) |
| The compressed loss puts 12 times more weight above 16 kHz (4.9 % vs 0.4 %), where the codec left almost nothing | **measured but untested**: still the candidate |

The test for the last one would be the same loss, cut above 16 kHz. I only tried c = 0.3, so the conclusion is "c = 0.3 in the loss hurts this model", not "compression never helps".

## After the re-run (1000 epochs)

`power_loss` at 1000 epochs with augmentation ([11](11_final.md)): **−1.00 dB [−1.21, −0.90]** on test, better on 1 track of 50 (−1.28 dB on validation, 1 seed). `power_in` and `norm` were not re-run.

```bash
uv run python -m src.train experiment=stereo_power_loss     # also stereo_power_in, stereo_power_both, stereo_norm
uv run python -m analysis.input_amplitude
uv run python -m analysis.mask_stats --ckpt reference=<ckpt> --ckpt power_loss=<ckpt>
```

## References

- S. Braun, I. Tashev, *A consolidated view of loss functions for supervised deep learning-based speech enhancement*, [arXiv:2009.12286](https://arxiv.org/abs/2009.12286), 2020 (venue not checked).
- K. Wilson et al., *Exploring Tradeoffs in Models for Low-latency Speech Enhancement*, [arXiv:1811.07030](https://arxiv.org/abs/1811.07030), 2018 (venue not checked).
