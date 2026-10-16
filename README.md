# Singing voice separation with a small U-Net

<!-- AUTHOR: rewrite this introduction in your own words (what you wanted to find out, what surprised you). The text below is a draft. -->
I reimplemented the U-Net of [Jansson et al. (ISMIR 2017)](https://archives.ismir.net/ismir2017/paper/000171.pdf) to separate the voice from the accompaniment, then used it as a testbed: take an idea from a more recent paper, say what I expect it to change, test it against a fixed baseline, and keep the negative results.

**The constraint is my laptop GPU (GTX 1650 Max-Q, 4 GB of VRAM).** Every model has to fit in it. The question is what a model that small can get out of the ideas of the last years.

Data: MUSDB18 (compressed stems), 84 train / 16 validation / 50 test tracks. Metric: stereo museval SDR, median over the 50 test tracks, with 95 % bootstrap intervals over tracks.

## Listen

[Demo page](https://gabrielplayeur.github.io/SingingVoiceSeparation/demo/): two test tracks, with the best, a typical and the worst 10 s excerpt picked by a fixed rule, so a good passage does not hide a bad one.

## Results

![stereo baseline against the ideal-mask oracle](figures/stereo_vs_oracle.png)

The baseline (9.8 M parameters, stereo) reaches **4.50 dB**. The ideal mask with the same reconstruction reaches 9.09 dB, so the model gets 49 % of what a magnitude mask can give and about 4.6 dB are left in the mask estimate itself, before the mixture phase matters.

| # | Question | At 150 epochs, 1 seed (test SDR, paired difference to the baseline) | At budget B, 3 seeds | Page |
|---|---|---|---|---|
| 1 | How far is the baseline from the oracle? | 4.50 against 9.09 dB; below the oracle on 50 of 50 tracks | | [01](experiments/01_baseline_vs_oracle.md) |
| 2 | Does stereo input help? | No: +0.04 dB [−0.18, +0.17] | | [02](experiments/02_mono_vs_stereo.md) |
| 3 | Is the learning rate badly tuned? | No: 1e-3 constant beats the lower rates and cosine by 0.15 to 0.68 dB (validation) | | [03](experiments/03_lr_schedule.md) |
| 4 | Do waveform losses help? | SI-SDR +0.11 dB and wSDR +0.12 dB, inside the noise; multi-resolution STFT −0.51 dB | | [04](experiments/04_waveform_loss.md) |
| 5 | Does compressing the magnitude help? | In the loss −0.82 dB; in the input +0.07 dB; per-crop normalisation −0.01 dB | | [05](experiments/05_amplitude_compression.md) |
| 6 | Is a wider model better? | w4 and w8 −0.85 and −0.81 dB; w32 +0.22 dB; w64 +0.42 dB (over the 4 GB budget) | | [06](experiments/06_size_quality.md) |
| 7 | Does a longer training context help? | 2.96 s −0.35 dB; 8.90 s 0.00; 11.88 s +0.12 dB | | [07](experiments/07_context_length.md) |
| 8 | Can the model be stored in int8? | Yes: 4.49 against 4.50 dB, 4× smaller (9.9 MB), but 2.3× slower on my CPU | | [08](experiments/08_int8_quantization.md) |

I expected the waveform losses to gain 0.3 to 1 dB and got 0.1. I expected compressing the magnitudes to help, as it does in speech enhancement; in the loss it cost 0.8 dB. Width is the only lever that beat the noise.

**Caveats.**
- All 150-epoch runs stop while the validation score is still rising, so those comparisons also measure how fast a model converges. I am re-running them at a longer budget B, with data augmentation and three seeds, decided on validation only. The B column is empty until that is done.
- One run moves by about 0.3 dB between two trainings of the same baseline; paired differences stay within ±0.15 dB. That is the yardstick for the 150-epoch column.
- The numbers do not compare with published ones: compressed MUSDB18 instead of HQ, SDR computed on vocals plus accompaniment instead of the official mixture.

<!-- AUTHOR: conclusion to write once the B column is filled. -->

## Reproduce

Environment with [uv](https://github.com/astral-sh/uv), configs with Hydra, logs with W&B.

```bash
uv sync
uv run python -m src.train experiment=stereo                    # baseline
uv run python -m src.evaluate CKPT -o outputs/eval_stereo.json  # test set, stereo museval SDR
uv run python -m analysis.make_demo extract CKPT musdb18/processed/test/<track>   # demo excerpts
```

Each experiment page lists its own commands and the script that makes its figure. Server runs go through `deploy/run_queue.py`, one group of [deploy/queue.yaml](deploy/queue.yaml) at a time.

## Repository

```
src/          model, Lightning module, data, train / evaluate / inference, int8
src/configs/  Hydra configs, one file per experiment in experiment/
analysis/     one script per figure or number
experiments/  one page per experiment: hypothesis, protocol, result; protocol.md, ideas.md
deploy/       experiment queue for a rented GPU
demo/         static listening page
```

Papers are cited on the experiment pages where they are used. Base paper: A. Jansson, E. Humphrey, N. Montecchio, R. Bittner, A. Kumar, T. Weyde, *Singing Voice Separation with Deep U-Net Convolutional Networks*, ISMIR 2017.
