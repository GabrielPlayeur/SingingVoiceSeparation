# 04 · Waveform losses

[← back to README](../README.md)

**In short.** I expected a loss on the reconstructed waveform to beat the L1 on magnitudes by 0.3 to 1 dB. At 150 epochs, SI-SDR and wSDR gave about +0.1 dB, inside the noise, and the multi-resolution STFT loss lost 0.5 dB. At 1000 epochs, **wSDR becomes a real gain (+0.25 dB, 3 seeds)** and SI-SDR a real loss (−0.47 dB).

## Hypothesis

The L1 loss compares magnitudes, but the evaluation compares waveforms, after the inverse STFT with the mixture's phase. Training directly on the waveform (`_to_wave` is differentiable) should bring the optimisation closer to what I measure.

**Prediction:** +0.3 to 1 dB SDR. **If rejected:** the loss mismatch isn't what limits the model, or the speech results don't transfer to music.

## The three losses

ŷ = iSTFT(mask · |X|, mixture phase). Code in [src/utils/losses.py](../src/utils/losses.py).

| config | loss | source |
|---|---|---|
| baseline | L1 on magnitudes | Jansson et al. 2017 |
| sisdr | −SI-SDR(ŷ, y), the validation metric itself | Le Roux et al. 2018 (they define the metric; using it as a loss is my extrapolation) |
| wsdr | weighted SDR: a vocals term and an accompaniment term (ẑ = x − ŷ); the second makes it sensitive to the output level | Choi et al. 2019, eq. 7 and 9 |
| mrstft | spectral convergence + log magnitude, at 3 resolutions | Yamamoto et al. 2020 |

None of these papers is about music separation: Choi and Le Roux work on speech, Yamamoto on a vocoder. Yamamoto's resolutions are for 24 kHz; I converted them to 44.1 kHz keeping the durations, which is my own adaptation. All fit in 4 GB (mrstft: 2.38 GiB with recomputation).

## Results (150 epochs, one seed)

![waveform losses](../figures/loss_study.png)

Each dot is a test track, variant minus baseline. The grey columns are two other trainings of the *same* baseline: their paired differences stay within about ±0.15 dB, which is the noise to beat.

| | val SI-SDR (last 10 ep.) | test Δ SDR vs baseline | better on |
|---|---|---|---|
| baseline | 3.85 dB | (4.47 dB) | |
| *same baseline, retrained* | *3.95 dB* | *+0.06 [−0.07, +0.15]* | *30/50* |
| sisdr | 3.92 dB | +0.11 [+0.01, +0.31] | 32/50 |
| wsdr | **4.16 dB** | +0.12 [−0.10, +0.36] | 29/50 |
| mrstft | 3.21 dB | **−0.51 [−0.71, −0.41]** | 10/50 |

- **SI-SDR and wSDR: +0.1 dB, the size of the rerun noise.** wSDR had the best validation curve of any w16 model, so I kept it as the one candidate to retest.
- **mrstft is clearly worse and unstable.** Its validation collapses by 4 dB from one epoch to the next while the training loss barely moves. Two untested causes: its log term spends effort on the 23 % of near-empty bins above the AAC cutoff, and BatchNorm running statistics. This says "my MR-STFT setup doesn't work here", not "MR-STFT doesn't work".
- **All three collapse on *PR - Happy Daze*** (−15 to −16 dB), the near-silent track. These losses are normalised by the target's energy, and training crops without voice are filtered out, so they never learn to output silence.

## After the re-run (1000 epochs)

With augmentation, 1000 epochs ([11](11_final.md)):

| | validation | test, paired [95 % CI] | better on |
|---|---|---|---|
| **wSDR** (3 seeds vs 3) | +0.24 dB | **+0.25 dB [+0.13, +0.32]** | 39/50 |
| SI-SDR (1 seed) | −0.69 dB | **−0.47 dB [−0.60, −0.32]** | 5/50 |

- **wSDR: the candidate is confirmed** once the models are trained to their plateau.
- **SI-SDR gets worse**, even on its own metric. The output level is not the cause: its scale cost is 0.11 dB, against 0.14 dB for the reference ([output_scale_1000ep.json](../figures/output_scale_1000ep.json)). ⚠ No tested explanation.
- Both still collapse on the two near-silent tracks.
- mrstft was not re-run (it failed by instability, not by lack of training). **wSDR was not tested at width 32**, so the final model keeps the L1 loss.

## Is the output level a risk?

SI-SDR ignores the output level, while SDR doesn't. On the 150-epoch baseline (80 validation crops), the estimated voice is slightly too loud (optimal gain −0.77 dB), and fixing it would gain only 0.13 dB.

![output level](../figures/output_scale.png)

```bash
uv run python -m src.train experiment=stereo_wsdr        # also stereo_sisdr, stereo_mrstft
uv run python -m analysis.loss_study --dir outputs/server/loss-v1 --rerun rerun_local=outputs/eval_stereo.json --rerun rerun_w16=outputs/server/size-v1/w16_s42/eval.json
uv run python -m analysis.output_scale --ckpt baseline=<ckpt> --ckpt sisdr=<ckpt>
```

## References

- J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, *SDR – half-baked or well done?*, [arXiv:1811.02508](https://arxiv.org/abs/1811.02508), 2018 (ICASSP 2019 from memory, not checked).
- H.-S. Choi, J.-H. Kim, J. Huh, A. Kim, J.-W. Ha, K. Lee, *Phase-aware Speech Enhancement with Deep Complex U-Net*, ICLR 2019, [arXiv:1903.03107](https://arxiv.org/abs/1903.03107).
- R. Yamamoto, E. Song, J.-M. Kim, *Parallel WaveGAN*, ICASSP 2020, [arXiv:1910.11480](https://arxiv.org/abs/1910.11480). Multi-resolution loss section and table 1.
