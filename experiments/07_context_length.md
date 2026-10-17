# 07 · Context length

[← back to README](../README.md)

**In short.** HT Demucs gains about 0.5 dB by training on longer excerpts. Here, shorter crops (2.96 s) hurt (−0.35 dB), and longer ones (8.90 s, 11.88 s) give nothing beyond the noise. That fits the U-Net's receptive field of 4.42 s: beyond it, a longer crop gives no output frame more context. Not re-run at 1000 epochs, since the argument doesn't depend on the budget.

## Why test it

The 5.93 s crop came with the first implementation, without a reason. Rouard et al. 2022 (HT Demucs, arXiv:2211.08553, Table 2) gain **+0.53 dB** going from 3.4 s to 7.8 s segments. The "+0.4 dB more at 12.2 s" quoted in their conclusion mixes several changes at once; the clean number in their tables is at most +0.14 dB. Their model is a Transformer that attends over the whole segment, so a longer segment really gives it more context. Mine is a plain U-Net.

## The receptive field (measured)

I back-propagated from one output frame to the input (`analysis.receptive_field`):

- **Total receptive field: 4.42 s** (381 frames). No output frame depends on audio further than ±2.2 s.
- Effective receptive field (where the gradient is not tiny): about 0.5 s, on an untrained network, so an order of magnitude only.

So a longer crop can only reduce **edge effects**: the share of output frames whose context runs past a crop edge goes from 100 % at 2.96 s to 74 % at 5.93 s and 37 % at 11.88 s. At inference, overlapping windows already hide most edges.

**Prediction:** 2.96 s worse by 0.2 to 0.5 dB, 8.90 and 11.88 s at most +0.1 to +0.2 dB.

## Setup

Only `data.crop_length` changes. The lengths are multiples of 64 STFT frames (2.96, 5.93, 8.90, 11.88 s), as the U-Net requires. A longer crop at batch 8 also means more audio per step, so a control `ctx12_b4` uses 11.88 s at batch 4: same audio per step as the baseline. All models are evaluated with the same 5.93 s windows.

## Results (150 epochs, one seed)

![context length](../figures/context_length.png)

| training crop | batch | Δ SDR vs baseline, paired [95 % CI] | better on |
|---|---|---|---|
| 2.96 s | 8 | **−0.35 [−0.48, −0.15]** | 13/50 |
| **5.93 s (baseline)** | 8 | | |
| 8.90 s | 8 | 0.00 [−0.12, +0.14] | 25/50 |
| 11.88 s | 8 | +0.12 [+0.03, +0.20] | 33/50 |
| 11.88 s, control | 4 | **−0.27 [−0.35, −0.19]** | 12/50 |
| *same baseline, retrained* | 8 | *+0.06 [−0.07, +0.15]* | *30/50* |

- **Shorter hurts, longer doesn't help.** +0.12 dB at 11.88 s is within what a retrained baseline gives.
- **At equal audio per step, the long crop is worse** (−0.27 dB). So the small gain of `ctx12` doesn't come from context. The control also halves the batch (noisier gradients, BatchNorm over 4 examples), so I can't say which of these causes the drop.
- **For the "attention at the bottleneck" idea**: more context doesn't help this network, so attention can't be justified by context alone.

## What I conclude

I keep 5.93 s. 11.88 s costs 30 % more time and twice the VRAM for no measurable gain. Not done: evaluating each model with windows of its own training length.

```bash
uv run python -m src.train experiment=stereo_ctx12        # also stereo_ctx3, stereo_ctx9, stereo_ctx12_b4
uv run python -m analysis.receptive_field
```

## References

- S. Rouard, F. Massa, A. Défossez, *Hybrid Transformers for Music Source Separation*, [arXiv:2211.08553](https://arxiv.org/abs/2211.08553), 2022 (ICASSP 2023 from memory, not checked). § 5.1 to 5.5 and Table 2 read.
