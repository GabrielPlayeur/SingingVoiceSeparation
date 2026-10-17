# 06 · Size/quality curve

[← back to README](../README.md)

**In short.** Width was the first change that clearly helped. At 150 epochs, w32 (39 M parameters, fits in 4 GB) beat w16 by +0.22 dB. At 1000 epochs with augmentation, the gap grows to **+0.34 dB** (3 seeds), and **w32 became the final model**.

## Hypothesis

At a fixed budget, a wider U-Net separates better, with diminishing returns.

**Prediction:** w4 about 1 dB below w16, w32 a few tenths above. **If rejected:** capacity isn't the limit, and gains must come from elsewhere.

## Setup

Only `model.base_channels` changes. Each doubling of width multiplies the parameters by 4. I don't vary depth: it is tied to the crop length, so it would change two things at once.

| config | parameters | fp32 weights | peak VRAM (1 step, GTX 1650) |
|---|---|---|---|
| w4 | 0.62 M | 2.3 MiB | 0.50 GiB |
| w8 | 2.46 M | 9.4 MiB | 0.74 GiB |
| **w16 (baseline)** | 9.82 M | 37.5 MiB | 1.28 GiB |
| w32 | 39.28 M | 149.9 MiB | 2.63 GiB |
| w64 | 157.11 M | 599.3 MiB | **6.02 GiB** (over 4 GB) |

w64 was trained on the rented GPU only, to see whether the curve keeps rising. It's not a candidate.

## Results (150 epochs, one seed)

![size / quality](../figures/size_quality.png)

The middle panel is the one to read: paired per-track differences against w16. The hollow points are two other trainings of w16 (+0.01 and −0.01 dB), the noise to beat.

| width | test SDR | Δ vs w16, paired [95 % CI] | better on | epochs run |
|---|---|---|---|---|
| w4 | 3.61 dB | **−0.85 [−1.20, −0.69]** | 4/50 | 150 |
| w8 | 3.70 dB | **−0.81 [−1.00, −0.63]** | 3/50 | 122, early stop |
| **w16** | 4.78 dB | | | 150 |
| w32 | 4.89 dB | **+0.22 [+0.13, +0.32]** | 35/50 | 138, early stop |
| w64 | 5.07 dB | **+0.42 [+0.30, +0.54]** | 44/50 | 150 |

- **Bigger is better above w16**, about +0.2 dB per doubling, with no sign of diminishing returns up to 157 M parameters.
- **Below w16 there is a cliff, not a slope**: w4 and w8 are about the same, then w16 jumps by 0.8 dB. I don't know why.
- **Early stopping cut w32 on a lucky peak.** Its validation is very noisy (1.11 dB from one epoch to the next), and it stopped while still improving. So its gain was underestimated. I should have disabled early stopping here; the re-run does.
- **Three trainings of w16 with the same seed** give medians of 4.47, 4.50 and 4.78 dB, but paired differences within ±0.15 dB. A single median moves a lot; paired differences don't. That's why every page compares paired differences.

## After the re-run (1000 epochs)

w32 against w16, both with augmentation, 3 seeds each ([11](11_final.md)):

- **+0.34 dB [+0.25, +0.40] on test**, better on 44/50 tracks (+0.46 dB on validation).
- On validation, the gap **grows** with training: +0.27 dB at epoch 150, +0.46 at 1000. With augmentation, the extra width is really used; it's not just a faster start.
- Without augmentation, the probe shows the opposite (+0.35 → +0.26 dB, one seed, [09](09_convergence_probe.md)): w32 saturates earlier.
- **w32 is the final model**: 5.89 dB on test, 2.63 GiB peak VRAM.

w4, w8 and w64 were not re-run.

```bash
uv run python -m src.train experiment=stereo_w32          # also stereo_w4, w8, w64
uv run python -m analysis.size_quality --run 16=outputs/server/size-v1/w16_s42/eval.json --run 32=outputs/server/size-v1/w32_s42/eval.json   # ...one --run per width
```
