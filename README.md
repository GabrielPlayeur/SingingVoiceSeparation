# Singing voice separation with a small U-Net

<!-- AUTHOR: rewrite this introduction in your own words (what you wanted to find out, what surprised you). The text below is a draft. -->
I reimplemented the U-Net of [Jansson et al. (ISMIR 2017)](https://archives.ismir.net/ismir2017/paper/000171.pdf) to separate the voice from the accompaniment, then used it as a testbed to take an idea from a more recent paper. 

**The constraint is my laptop GPU (GTX 1650 Max-Q, 4 GB of VRAM).** Every model has to fit in it. The question is what a model that small can get out of the ideas of the last years and see how new techniques and tricks can improve the output of the model.

Data: MUSDB18 (compressed stems), 84 train / 16 validation / 50 test tracks.\
Metric: stereo museval SDR, median over the 50 test tracks, with 95 % bootstrap intervals over tracks.

## Listen

[Demo page](https://gabrielplayeur.github.io/SingingVoiceSeparation/demo/): the final model on the only two test tracks whose licence allows publishing them, with the best and the worst 10 s excerpt picked by a fixed rule. They happen to be the two hardest tracks for it (see the page).

## Results

| model (test SDR, official mixture, 50 tracks) | median [95 % CI] | |
|---|---|---|
| Baseline: Jansson-style U-Net, 9.8 M parameters, 150 epochs | 4.47 [3.54, 5.37] | starting point |
| Same U-Net, 1000 epochs, data augmentation | 5.45 [3.89, 5.95] | 3 seeds |
| **Final: twice as wide (39 M parameters, 2.6 GB of VRAM), 1000 epochs, augmentation** | **5.89 [4.29, 6.30]** | 3 seeds; +0.99 dB over the baseline, better on 48/50 tracks |
| [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) `umx`, run through the same pipeline | 6.43 [5.21, 7.24] | published: 6.32 dB |
| Ideal magnitude mask (oracle), same reconstruction as the U-Net | 9.09 | |

Running Open-Unmix through my evaluation gives back its published score (6.43 against 6.32 dB), so the numbers are on the same scale as the literature for compressed MUSDB18 and can give us a comparaison point for the final model resultat that is 0.55 dB below Open-Unmix.

**The main finding is about the method, not a trick.** My first eight experiments ran for 150 epochs. A long probe ([09](experiments/09_convergence_probe.md)) showed every model was still improving there (+0.6 dB more by epoch 1000), so I re-ran what mattered at 1000 epochs, with 3 seeds where a decision depended on it.

| # | Question | At 150 epochs, 1 seed (test SDR, paired difference to the baseline) | At 1000 epochs, with augmentation (test, paired difference) | Page |
|---|---|---|---|---|
| 1 | How far is the baseline from the oracle? | 4.50 against 9.09 dB; below the oracle on 50 of 50 tracks | | [01](experiments/01_baseline_vs_oracle.md) |
| 2 | Does stereo input help? | No: +0.04 dB [−0.18, +0.17] | Still no: stereo − mono −0.15 dB [−0.28, +0.14] (1 mono seed) | [02](experiments/02_mono_vs_stereo.md), [11](experiments/11_final.md) |
| 3 | Is the learning rate badly tuned? | No: 1e-3 constant beats the lower rates and cosine (validation) | A decay at the end halves the end-of-run noise but gains nothing (+0.04 dB) | [03](experiments/03_lr_schedule.md), [10](experiments/10_recipe.md) |
| 4 | Do waveform losses help? | SI-SDR +0.11 dB and wSDR +0.12 dB, inside the noise; multi-resolution STFT −0.51 dB | **wSDR +0.25 dB** [+0.13, +0.32] (3 seeds); SI-SDR −0.47 dB | [04](experiments/04_waveform_loss.md), [11](experiments/11_final.md) |
| 5 | Does compressing the magnitude help? | In the loss −0.82 dB; in the input +0.07 dB | In the loss −1.00 dB: not a convergence effect | [05](experiments/05_amplitude_compression.md), [11](experiments/11_final.md) |
| 6 | Is a wider model better? | w32 +0.22 dB; w64 +0.42 dB (over the 4 GB budget) | **w32 +0.34 dB** [+0.25, +0.40] (3 seeds); the gap grows with training | [06](experiments/06_size_quality.md), [11](experiments/11_final.md) |
| 7 | Does a longer training context help? | 2.96 s −0.35 dB; 8.90 s 0.00; 11.88 s +0.12 dB | not re-run (bounded by the 4.4 s receptive field) | [07](experiments/07_context_length.md) |
| 8 | Can the model be stored in int8? | Yes: 4.49 against 4.50 dB, 4× smaller (9.9 MB), but 2.3× slower on my CPU | not re-run on the final model | [08](experiments/08_int8_quantization.md) |
| 9 | Does data augmentation help? | (not tested) | **+0.19 to +0.30 dB** (3 seeds per cell); −0.4 to −0.7 dB at 150 epochs | [09](experiments/09_convergence_probe.md), [10](experiments/10_recipe.md) |
| 10 | Was the head bug costly? | (no clean measurement) | No: −0.01 dB; the trained head learned around it | [01](experiments/01_baseline_vs_oracle.md), [11](experiments/11_final.md) |

![each change against its reference, test](figures/final_test.png)

All the validation decisions held on test, with the same sign. Training length, augmentation and width did most of the work; of the ideas borrowed from speech papers, only the weighted-SDR loss helped, and compressing magnitudes in the loss hurt by 1 dB.

**Caveats.**
- The seed-to-seed spread is 0.09 dB on the validation score (3 seeds × 4 configurations). One-seed rows need a difference of about 0.2 dB to mean anything; the CIs above only cover track-to-track variability.
- The 150-epoch column builds the mixture as vocals plus accompaniment; the 1000-epoch column and the first table use the official mixture. The two differ by 0.03 dB on the baseline (4.50 against 4.47 dB).
- wSDR and width 32 were each tested against the w16 recipe, not together: the final model uses the L1 loss.
- Compressed MUSDB18, not HQ: only comparable with results on the same version.

<!-- AUTHOR: conclusion to write once the B column is filled. -->

## Reproduce

Environment with [uv](https://github.com/astral-sh/uv), configs with Hydra, logs with W&B.

```bash
uv sync
uv run python -m src.train experiment=stereo                    # 150-epoch baseline
uv run python -m src.train experiment=recipe model.base_channels=32   # final model: w32, 1000 epochs, augmentation (2.6 GiB of VRAM)
uv run python -m src.evaluate CKPT -o outputs/eval_stereo.json  # test set, stereo museval SDR
uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_umx --niter 1 -o outputs/umx_niter1.json   # calibration
uv run python -m analysis.make_demo extract CKPT musdb18/processed/test/<track>   # demo excerpts
uv run python -m analysis.final_test                            # every test comparison of the re-run (figures/final_test.*)
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
