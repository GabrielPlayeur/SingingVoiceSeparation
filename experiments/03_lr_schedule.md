# 03 · Learning rate and scheduler

[← back to README](../README.md)

**In short.** Nothing beat a constant 1e-3: lower rates and cosine decay were all worse at 150 epochs. The more telling finding was that "higher is better" and the still-rising curve meant the baseline was **under-trained**, which led to the re-run ([09](09_convergence_probe.md)). At 1000 epochs, a decay at the end of training halves the end-of-run noise but gains nothing on test.

## Why I suspected the setting

- 1e-3 isn't sourced: it's Adam's default, and Jansson et al. give no value.
- The papers I read use lower rates (Braun & Tashev 1e-4, Wilson et al. 2.1e-4, Yamamoto et al. 1e-4), but with other networks and optimisers.
- The baseline's validation curve was still rising at epoch 150 (+0.6 dB per 100 epochs) and noisy: 0.32 dB from one epoch to the next. A constant 1e-3 was one possible cause of the noise.

**Hypothesis.** A lower rate, or a decay over the 150 epochs, gives a stabler curve and a better model.

## Setup

Stereo baseline. Only the rate and the scheduler change. Cosine decay without restarts, to 0, updated every step (equation 5 of Loshchilov & Hutter, SGDR, ICLR 2017; I only take the decay). Early stopping off for the cosine runs, since it would cut the decay.

**Decision rule, written before the runs:** decide on validation only. Score = mean of `val/si_sdr_median` over the last 10 epochs, not the max (the max rewards a lucky epoch).

## Results

![learning rate study](../figures/lr_schedule.png)

| config | val SI-SDR, last 10 epochs | best epoch |
|---|---|---|
| **1e-3 constant (baseline)** | **3.95 ± 0.08 dB** | 4.14 dB (ep. 134) |
| 1e-3 cosine | 3.80 ± 0.03 dB | 3.88 dB |
| 3e-4 constant | 3.73 ± 0.05 dB | 3.84 dB |
| 3e-4 cosine | 3.47 ± 0.03 dB | 3.56 dB |
| 1e-4 constant (early stop at 129) | 3.28 ± 0.11 dB | 3.46 dB |

± is the standard deviation over the last 10 epochs. On test (not used for the choice), the baseline also beats the three lower-rate runs: 4.50 dB against 4.36 to 4.05 dB.

## What I conclude

- **Hypothesis rejected.** 1e-3 constant beats the others by 0.15 to 0.68 dB. I keep it.
- **Higher is better** for constant rates (1e-4 < 3e-4 < 1e-3). Together with a curve still rising at epoch 150, this points to under-training, not to a rate that is too high.
- **Cosine lowers the noise** (0.03 vs 0.08 dB) but costs 0.15 dB: it slows learning while the model is still improving.
- The best epoch (4.14 dB) sits 0.2 dB above the mean of the last 10 (3.95 dB). Keeping the "best" checkpoint picks noise.
- Limits: one seed; the grid stops at 1e-3.

## After the re-run (1000 epochs)

In [10](10_recipe.md) (3 seeds per cell), I tested a different decay: constant rate, then linear to 0 over the last 20 % (`wsd`). It gains +0.13 dB on validation, below the decision threshold (0.15 dB), and +0.04 / −0.01 dB on test. It does halve the noise of the last epochs (0.04 against 0.09–0.10 dB). The constant 1e-3 stays.

```bash
uv run python -m src.train experiment=stereo_lr3e4      # also stereo_cosine, stereo_lr3e4_cosine, stereo_lr1e4
uv run python -m analysis.lr_schedule --run stereo=fgvzjxua --run cosine=543aaoll --run lr3e4=<id> --run lr3e4_cosine=<id> --run lr1e4=<id>
```

## References

- I. Loshchilov, F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017, [arXiv:1608.03983](https://arxiv.org/abs/1608.03983).
- Learning rates of the other papers: Braun & Tashev, Wilson et al. ([05](05_amplitude_compression.md#references)), Yamamoto et al. ([04](04_waveform_loss.md#references)), read in their training sections.
