# 01 · Baseline vs oracle

[← back to README](../README.md)

**In short.** The 150-epoch baseline scores **4.50 dB** SDR, about half of the oracle mask (9.09 dB). Run through my own pipeline, Open-Unmix gets 6.43 dB, against 6.32 dB published, so the evaluation can be trusted. After the re-run at 1000 epochs, the final model reaches **5.89 dB** ([11](11_final.md)).

## The model

Jansson et al., *Singing Voice Separation with Deep U-Net Convolutional Networks*, ISMIR 2017. Code: [src/models/unet.py](../src/models/unet.py), [src/module.py](../src/module.py).

- **Input:** magnitude STFT of a 5.93 s stereo crop (44.1 kHz, window 2048, hop 512), shape (batch, 2, 1025, 512). No normalisation.
- **U-Net:** 6 encoder blocks (conv 5×5 stride 2, BatchNorm, LeakyReLU 0.2, 16 → 512 channels), 6 decoder blocks (transposed conv, BatchNorm, ReLU) with skip connections, `Dropout2d(0.5)` on the first 3 decoders. A 1×1 conv and a sigmoid give one mask per channel. 9.8 M parameters.
- **Reconstruction:** vocals = mask × |mixture| with the mixture's phase, then inverse STFT. Accompaniment = mixture − vocals.
- **Training:** L1 on magnitudes, Adam 1e-3, 150 epochs of 500 random crops (batch 8), seed 42.
- **Evaluation:** museval v4 SDR on the 50 test tracks (1 s windows, median per track, then median over tracks). Full tracks are processed in 5.93 s windows with 50 % overlap.

## Differences from the paper

| | Jansson et al. 2017 | Here |
|---|---|---|
| Training data | ~20,000 tracks, private | MUSDB18 compressed, 84 tracks |
| Sample rate | 8,192 Hz | 44,100 Hz |
| STFT window / hop | 1024 / 768 | 2048 / 512 |
| Magnitude normalisation | scaled to [0, 1] | none |
| Channels | mono | stereo (2 masks) |
| Models | two U-Nets (vocals, instrumental) | one; accompaniment = mixture − vocals |
| Dropout | 50 % | `Dropout2d` (whole channels) |
| Evaluation | iKala, MedleyDB, BSS Eval v3 | MUSDB18 test, museval v4 |

The paper gives no learning rate, so 1e-3 is Adam's default; [03](03_lr_schedule.md) tests it.

## The head bug

My first version kept BatchNorm + ReLU in the last decoder block, before the 1×1 conv and the sigmoid. The mask was then `sigmoid(w·r + b)` with `r ≥ 0`. In the checkpoint trained this way, all head weights were negative, so the mask never went above **0.727** and sat exactly there on **46 %** of the bins. Fixed in `68ba08f`.

At 1000 epochs, a clean rerun with the old head costs **−0.01 dB [−0.13, +0.12]** ([11](11_final.md)). That run learned one positive weight per channel ([[−1.06, +1.48], [−1.07, +1.48]]), so its mask was not capped. The ceiling was a property of one checkpoint, not of the architecture. I keep the fix, since it removes the failure mode.

## Results

The **oracle** is the ideal ratio mask |V|/|X| clipped to [0, 1], with the same reconstruction as the model. It shows what a mask of this kind can reach; it is not a strict ceiling (a phase-aware mask does better).

| median over 50 test tracks [95 % CI] | model | oracle |
|---|---|---|
| SDR | **4.50 dB** [3.56, 5.38] | **9.09 dB** [8.25, 10.31] |
| SI-SDRi | 9.31 dB [8.41, 10.65] | 15.49 dB [14.52, 17.04] |

![stereo vs oracle](../figures/stereo_vs_oracle.png)

- The model is below the oracle on all 50 tracks. The paired gap is **4.59 dB** [4.11, 4.79], and it's fairly regular (mostly 3 to 6 dB).
- Difficulty comes mostly from the track: model and oracle ranks correlate at 0.86 (Spearman).
- The extreme case is *PR - Happy Daze* (model −27.7 dB, oracle ~0 dB), whose voice is almost silent. SDR means little there.
- SDR and SI-SDRi aren't on the same scale: SI-SDRi is a gain over the mixture (which starts around −4 dB), SDR is absolute.

![by genre](../figures/stereo_by_genre.png)

By genre, SDR goes from 6.91 dB (Rap, 3 tracks) to −1.85 dB (Electronic, 4 tracks, pulled down by the two *PR* tracks). With 2 to 4 tracks per genre outside Pop/Rock, none of these differences is significant.

## Calibration against Open-Unmix

I ran the pretrained Open-Unmix `umx` (Stöter et al., JOSS 2019) through the same evaluation ([src/evaluate_umx.py](../src/evaluate_umx.py)): same 50 tracks, official `mixture.wav`, stereo museval v4.

| model (official mixture) | SDR [95 % CI] | paired vs baseline |
|---|---|---|
| my baseline (150 epochs) | 4.47 dB [3.54, 5.37] | |
| Open-Unmix without Wiener (magnitude mask + mixture phase, like mine) | 6.06 dB [5.32, 7.30] | +1.44 dB, better on 43/50 |
| Open-Unmix with Wiener (published setup) | **6.43 dB** [5.21, 7.24] | +1.69 dB, better on 43/50 |
| final model, 1000 epochs ([11](11_final.md)) | **5.89 dB** [4.29, 6.30] | +0.99 dB, better on 48/50 |

- The published `umx` score is 6.32 dB. ⚠ I took it from Pac-HuBERT (arXiv:2304.02160) and Papers with Code, not from the Open-Unmix paper. My 6.43 dB is well inside the CI, so the pipeline reproduces it.
- Most of the baseline's gap to Open-Unmix (1.44 of the 1.69 dB) was in the network and its training, not in the Wiener filter. The re-run cut it to 0.27 dB at equal reconstruction.
- This is still compressed MUSDB18: not comparable with results on HQ.

The original mono model gave the same picture (4.70 vs 9.21 dB under the mono protocol); I switched to stereo after [02](02_mono_vs_stereo.md).

```bash
uv run python -m analysis.baseline_vs_oracle outputs/eval_stereo.json --tag stereo
uv run python -m src.evaluate CKPT --mixture official --no-oracle -o outputs/eval_stereo_official.json
uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_umx --niter 1 -o outputs/umx_niter1.json   # --niter 0: no Wiener
```
