# Singing Voice Separation with a small U-Net

**Question.** How far can a small U-Net, sized to train in 4 GB of VRAM, get on singing voice separation, and which ideas from more recent papers actually help it?
**Protocol.** MUSDB18 (compressed stems), 84 train / 16 val / 50 test tracks. Metric: stereo museval SDR, median over the 50 test tracks. One change at a time against a fixed baseline. **One training seed per run so far** (3-seed runs are set up but not run yet, see [Next steps](#next-steps)). Models are sized for a 4 GB laptop GPU; some runs were trained on a rented GPU to go faster.
**Key numbers.** Baseline (Jansson et al. 2017 U-Net, stereo): **4.50 dB SDR**. Ideal-mask oracle with the same reconstruction: **9.09 dB**. Stereo vs mono input: **+0.04 dB** (95% CI [−0.18, +0.17]).
**What I learned so far.**
1. The model gets about half of the oracle's SDR, and the gap is fairly regular across tracks: most of what's left to gain is in estimating the mask, not in the phase.
2. Feeding the model stereo instead of mono doesn't help, even though the voice is clearly more centred than the accompaniment in the data.
3. Training tweaks haven't paid off yet. A constant 1e-3 learning rate beats lower ones and cosine decay (the baseline is under-trained, not unstable). Waveform losses give at most +0.1 dB, about the gap between two trainings of the same model, and compressing magnitudes in the loss (`|X|^0.3`) costs 1.1 dB on validation.

**Status.** 4 experiments done (oracle, stereo, learning rate, waveform losses), amplitude compression trained with validation results only (test pending). Next: 3 seeds of the baseline, size/quality curve.

---

## Why this project

I started by reimplementing the U-Net of [Jansson et al. (ISMIR 2017)](https://archives.ismir.net/ismir2017/paper/000171.pdf), one of the first deep U-Nets for voice separation. Then I wanted to turn it into a small research project: take ideas from more recent papers, state what I expect each one to change, test it against the baseline, and keep the negative results too.

The constraint that ties it together is my laptop GPU: a GTX 1650 Max-Q with **4 GB of VRAM**. Every model has to fit in that budget. The question is what a model that small can borrow from the state of the art.

**Where training actually runs.** All models are sized for 4 GB. The baseline peaks at 1.37 GiB during training (measured on the GTX 1650). But to get through experiments faster, I also rent a GPU (an A800 80 GB on AutoDL) and train several runs in parallel there. The 4 GB rule is about the model, not the machine: server runs aren't capped, so it's on me to measure peak VRAM locally before adding a model. So far the baseline, the mono model and the `cosine` run were trained locally, and the `lr3e4`, `lr3e4_cosine` and `lr1e4` runs on the server. Same code, data, split and seed. I trained two configs on both machines with the same seed: `lr3e4_cosine` ended 0.05 dB apart on validation, and the baseline 0.10 dB apart on validation and +0.06 dB [−0.07, +0.15] apart on test SDR. Until the 3-seed runs are done, that ~0.1 dB is my best guess at the noise between two trainings.

## The baseline

The model follows Jansson et al. 2017, with a few changes listed in [experiments/01](experiments/01_baseline_vs_oracle.md#differences-from-the-paper):

- Input: magnitude STFT of the mixture (44.1 kHz, 2048-point window, hop 512), one channel per audio channel.
- U-Net: 6 conv encoder blocks (5×5, stride 2, 16 → 512 channels), 6 transposed-conv decoder blocks with skip connections, 50% dropout on the first 3 decoders. 9.8 M parameters.
- Output: a mask in [0, 1] (sigmoid). Estimated vocals = mask × mixture magnitude, recombined with **the mixture's phase**, then inverse STFT.
- Training: L1 loss on magnitudes, Adam 1e-3, random ~5.9 s crops, 150 epochs of 500 crops, batch 8.

**A bug I found along the way.** In my first version, the last decoder block still had BatchNorm + ReLU before the final 1×1 conv and sigmoid. Since the ReLU output is ≥ 0, the mask was capped at `sigmoid(bias)` and stuck at exactly that value wherever the ReLU output was zero. In the checkpoint trained with this bug, the mask never went above 0.727 and sat exactly at 0.727 on 46% of time-frequency bins (measured). I removed the ReLU/BN from that block (commit `68ba08f`). I don't have a clean before/after score: the only pre-fix run with a test score used an older evaluation pipeline and stopped at 47 epochs, so comparing it to the current baseline would mix several changes. A controlled rerun with the old head is on the to-do list.

## Results so far

| # | Experiment | Hypothesis | Result (test SDR unless stated) | Verdict | Details |
|---|---|---|---|---|---|
| 1 | Baseline vs oracle | There is a lot of headroom before the mixture phase becomes the limit | 4.50 dB vs 9.09 dB; gap 4.59 dB [4.11, 4.79]; model below oracle on 50/50 tracks | Confirmed | [01](experiments/01_baseline_vs_oracle.md) |
| 2 | Mono → stereo input | Voice is more centred than the accompaniment, so stereo should help | +0.04 dB [−0.18, +0.17], p = 0.89 | Rejected (1 seed) | [02](experiments/02_mono_vs_stereo.md) |
| 3 | Learning rate / scheduler | A lower LR or cosine decay gives a stabler, better model | 1e-3 constant wins by 0.15 to 0.68 dB (validation) | Rejected | [03](experiments/03_lr_schedule.md) |
| 4 | Waveform losses | Training on the reconstructed waveform gives +0.3 to 1 dB | SI-SDR +0.11 dB, wSDR +0.12 dB, MR-STFT −0.51 dB [−0.71, −0.41] | Rejected (1 seed) | [04](experiments/04_waveform_loss.md) |
| 5 | Amplitude compression | `\|X\|^0.3` gives more weight to quiet bins and helps, mostly in the loss | **Validation only**: in the loss −1.11 dB, input only −0.18 dB, per-crop norm −0.05 dB | Rejected for the loss (1 seed, test pending) | [05](experiments/05_amplitude_compression.md) |

All CIs are 95% bootstrap intervals over the 50 test tracks. They only capture track-to-track variability, not seed-to-seed variability, which I haven't measured yet (two trainings of the same baseline differ by about 0.1 dB).

**These numbers aren't comparable to published results.** I evaluate on vocals + accompaniment instead of the official (compressed) mixture, I use compressed MUSDB18 rather than MUSDB18-HQ, and I haven't checked my evaluation against a published baseline. I read them as internal comparisons only.

### 1. How far is the baseline from its ceiling?

The oracle is the ideal mask, clipped to [0, 1] and applied to the mixture with the mixture's phase. It's the same reconstruction as the model, so it's the best this *kind* of model could do, not an absolute ceiling.

![stereo baseline vs oracle](figures/stereo_vs_oracle.png)

- The model reaches **49%** of the oracle's median SDR (4.50 vs 9.09 dB).
- Track difficulty is shared: the rank correlation between model and oracle SDR is 0.86 (Spearman). Tracks that are hard for the model are already hard for the oracle.
- The gap is fairly regular (median 4.6 dB, most tracks between 3 and 6 dB). It looks like a general "model" loss more than a few broken tracks. The 27.7 dB outlier is *PR - Happy Daze*, where the voice is almost inaudible, so its SDR doesn't mean much.

So there are ~4.6 dB to gain on the mask itself before having to deal with the phase (complex input or complex mask).

### 2. Does stereo help?

I measured that the voice is much more centred than the accompaniment: between 400 Hz and 12.8 kHz its Side/Mid ratio is 9 to 12 dB lower. So I expected a 2-channel model to use that cue.

| SDR median, stereo protocol | mono model | stereo model | paired difference |
|---|---|---|---|
| SDR | 4.51 dB | 4.50 dB | +0.04 dB [−0.18, +0.17], p = 0.89 |
| SI-SDRi | 9.29 dB | 9.31 dB | +0.03 dB [−0.13, +0.31], p = 0.45 |

![mono vs stereo](figures/compare_mono_stereo_protocole_stereo.png)

No gain and no loss: stereo wins on 26 tracks out of 50. My best guess (not tested) is that the mono downmix *is* the Mid signal, so it already keeps the centred voice and drops part of the accompaniment. The mono model gets part of the cue for free. Also, the model only sees the magnitudes of L and R, not their phase difference. A Mid/Side input would give it the cue directly; that's a possible follow-up. I kept stereo as the baseline because it's the standard evaluation protocol. With one seed, this conclusion still needs the 3-seed runs to hold up.

### 3. Is the baseline badly tuned?

1e-3 is just Adam's default (the paper doesn't give a value), and the papers I read use lower values on other architectures. The baseline's validation curve was also noisy (0.32 dB between consecutive epochs). So I tested 3e-4, 1e-4 and cosine decay, picking on validation only.

![learning rate study](figures/lr_schedule.png)

| config | val SI-SDR, mean of last 10 epochs |
|---|---|
| **1e-3 constant (baseline)** | **3.95 ± 0.08 dB** |
| 1e-3 cosine | 3.80 ± 0.03 dB |
| 3e-4 constant | 3.73 ± 0.05 dB |
| 3e-4 cosine | 3.47 ± 0.03 dB |
| 1e-4 constant | 3.28 ± 0.11 dB (early-stopped at epoch 129) |

The higher the constant learning rate, the better, and the curve is still rising at epoch 150. That points to under-training, not to a learning rate that's too high. Cosine decay smooths the curve but costs 0.15 to 0.26 dB, probably because it shrinks the LR while the model is still learning. I kept 1e-3 constant. I didn't try higher than 1e-3, so a higher LR or more epochs might still do better.

### 4. Does training on the waveform help?

The L1 loss only looks at magnitudes, but I evaluate waveforms rebuilt with the mixture's phase. So I expected losses computed on the reconstructed waveform to gain +0.3 to 1 dB. I tried three, taken from speech papers: SI-SDR (Le Roux et al.), weighted-SDR (Choi et al.) and a multi-resolution STFT loss (Yamamoto et al.). All four runs, including a new baseline, were trained together on the server.

![waveform losses](figures/loss_study.png)

The "baseline rerun" column is the local baseline, same config and seed: it shows how much two trainings of the same model differ.

| loss | test SDR | Δ SDR vs baseline (paired) | better on | val SI-SDR (last 10 epochs) |
|---|---|---|---|---|
| L1 magnitude (baseline, server) | 4.47 dB | | | 3.85 ± 0.11 dB |
| SI-SDR | 4.71 dB | +0.11 [+0.01, +0.31] | 32/50 | 3.92 ± 0.08 dB |
| weighted-SDR | 4.63 dB | +0.12 [−0.10, +0.36] | 29/50 | 4.16 ± 0.09 dB |
| multi-resolution STFT | 3.88 dB | −0.51 [−0.71, −0.41] | 10/50 | 3.21 ± 0.13 dB |

- SI-SDR and wSDR gain about 0.1 dB, the same size as the gap between two trainings of the baseline (+0.06 dB). I can't call that a gain with one seed. wSDR has the best validation score so far, so it's the one to retest with seeds.
- All three waveform losses collapse on *PR - Happy Daze* (−15 to −16 dB), the track where the voice is almost inaudible. My guess: these losses are normalised by the target's energy, so they never push the model to output silence.
- The multi-resolution STFT loss is clearly worse, and its validation score drops by up to 4 dB from one epoch to the next while the training loss stays flat. That's my own 44.1 kHz adaptation of a speech-vocoder loss, so the conclusion is about this setup, not about MR-STFT in general.

I keep the L1 magnitude loss.

### 5. Does compressing the magnitude help?

The network sees raw magnitudes with 58 dB of dynamic range, and the linear L1 loss is dominated by loud low-frequency bins. Two speech-enhancement papers (Braun & Tashev, Wilson et al.) compress magnitudes with `|X|^0.3`, mostly in the loss. I tried it in the loss, in the input, in both, and a per-crop normalisation for volume invariance. **Only validation is available so far**: the test evaluation is still on the server.

![amplitude compression](figures/power_study.png)

| config | val SI-SDR (last 10 epochs) | Δ vs baseline |
|---|---|---|
| baseline (\|X\|, linear L1) | 3.85 ± 0.11 dB | |
| `\|X\|^0.3` in the loss | 2.74 ± 0.21 dB | −1.11 |
| `\|X\|^0.3` in the input | 3.67 ± 0.24 dB | −0.18 |
| both | 2.79 ± 0.10 dB | −1.06 |
| per-crop normalisation | 3.79 ± 0.13 dB | −0.05 |

- Compressing the loss is the part the papers supported, and it's the part that hurts: −1.1 dB, ten times the gap between two baseline trainings. I measured that the compressed target puts 12 times more weight above 16 kHz, where the AAC codec has cut most of the content (4.9% vs 0.4% of the total). I don't think that's the whole story. My other guess is that `x^0.3` is so steep near 0 that the loss pushes the mask to exactly zero wherever the voice is weak. I haven't tested it.
- Compressing only the input changes nothing measurable, and neither does normalisation. That's expected for `norm`: its job is volume robustness, which validation at the original level can't show. That test needs its checkpoint.

I drop compression in the loss; the rest waits for the test numbers.

## The data, in short

Details and figures in [experiments/00](experiments/00_dataset.md).

- **Official mixture vs vocals + accompaniment.** In compressed MUSDB18, the mixture isn't exactly the sum of the stems. The residual is −30.7 dB below the mixture (median) and mostly above 17 kHz. Applying the oracle mask to the real mixture instead of the sum changes SI-SDR by 0.05 dB, so I train and evaluate on the sum.
- **The test set is skewed:** 72% Pop/Rock, no Rock/Pop/Singer-Songwriter/Jazz, no MedleyDB tracks. Per-genre scores outside Pop/Rock rest on 2 to 4 tracks.
- **Split:** train/val split by artist (no shared artist). 3 artists appear in both train and test (3 test tracks out of 50).

## Next steps

Everything below is **planned, not measured yet**. Each page has the hypothesis, the prediction and the protocol, written before running.

0. **3 seeds of the baseline, mono and stereo** (`seeds` group in [deploy/queue.yaml](deploy/queue.yaml), seeds 42/43/44 on the same machine). Without this I can't say whether a 0.1 to 0.3 dB difference is real, including "stereo brings nothing".
1. **Finish amplitude compression**: bring back the test evaluations of `power-v1`, plus the `norm` checkpoint for the volume test. See [experiments/05](experiments/05_amplitude_compression.md#whats-left).
2. **Size/quality curve**: U-Net widths 4 to 64 (0.6 M to 157 M parameters). The other experiments get placed on this curve: is a trick worth more than just a bigger model? Plan in [experiments/06](experiments/06_size_quality.md).
3. **Context length**: train on crops of 2.96, 8.90 and 11.88 s instead of 5.93 s, plus a control with the same audio per step. The U-Net's receptive field is only 4.4 s, so I expect far less than the +0.5 dB HT Demucs gets from longer segments (Rouard et al. 2022). It also tells me whether attention at the bottleneck is worth trying. Plan in [experiments/07](experiments/07_context_length.md).
4. **Follow-ups on the losses**: output-level analysis of the SI-SDR and wSDR models (needs their checkpoints from the server), and wSDR with 3 seeds if the baseline's seed spread turns out to be well under 0.1 dB.
5. Smaller items: a controlled rerun of the head bug, training on the official mixture, int8 quantisation (size, latency, quality loss), a short audio demo.

## Ideas I'm not working on yet

These came out of the first training and my reading. They're further away, mostly because of the 4 GB budget or because they change several things at once.

- **Deal with the phase.** The mixture phase limits the model wherever the accompaniment dominates. Options: give the network the complex spectrogram, or estimate a complex mask whose magnitude can exceed 1, as in [Kong et al. (ISMIR 2021)](https://arxiv.org/abs/2109.05418), which decouples magnitude and phase estimation.
- **Better intermediate blocks**, like the TFC-TDF blocks of [Choi et al. (ISMIR 2020)](https://arxiv.org/abs/1912.02591).
- **Attention at the bottleneck.** After 6 stride-2 layers, each bottleneck position covers about 0.75 s × 1.4 kHz, so attention there would only see very coarse tokens. I'd need to try it to know whether that's useful.
- **A second stage** that predicts a residual to clean up what the mask missed, as in [TS-BSMamba2 (Bai et al., 2024)](https://arxiv.org/abs/2409.06245).
- **Masks for several sources with shared weights**, as in [Kadandale et al. (2020)](https://arxiv.org/abs/2003.10414), so the instruments "compete" with the voice for each bin.
- **A generative approach**, which would recreate a clean voice in bins shared with instruments instead of averaging them. Probably below state of the art, but a different trade-off.

## Reproduce

Environment managed with [uv](https://github.com/astral-sh/uv), configs with Hydra, logs with W&B (project `VoiceSeparation`).

```bash
uv sync
uv run python -m src.train experiment=stereo                    # baseline (stereo)
uv run python -m src.train data.channels=1                      # mono model
uv run python -m src.evaluate CKPT -o outputs/eval_stereo.json  # test set, stereo museval SDR
uv run python -m analysis.baseline_vs_oracle outputs/eval_stereo.json --tag stereo
uv run python -m analysis.compare_runs outputs/eval_stereo.json outputs/eval_<name>.json
uv run python -m analysis.loss_study                             # experiment 04, from outputs/server/loss-v1
# experiment 05: same script with --run NAME=DIR (see experiments/05)
```

Each experiment page lists its own commands. Server runs go through `deploy/run_queue.py` (one group of [deploy/queue.yaml](deploy/queue.yaml) at a time).

## Repository layout

```
src/            model (models/unet.py), Lightning module, data, train / evaluate / inference
src/configs/    Hydra configs; experiment/ holds one file per experiment
analysis/       scripts that produce every figure and number in this README
experiments/    one page per experiment: hypothesis, protocol, results, discussion
figures/        generated figures
deploy/         scripts to run the experiment queue on a rented GPU
```

## References

- A. Jansson, E. Humphrey, N. Montecchio, R. Bittner, A. Kumar, T. Weyde, *Singing Voice Separation with Deep U-Net Convolutional Networks*, ISMIR 2017.
- Other papers are cited on the experiment pages where they're used: Loshchilov & Hutter 2017 ([03](experiments/03_lr_schedule.md)); Le Roux et al. 2018, Choi et al. 2019, Yamamoto et al. 2020 ([04](experiments/04_waveform_loss.md)); Braun & Tashev 2020, Wilson et al. 2018 ([05](experiments/05_amplitude_compression.md)); Rouard et al. 2022 ([07](experiments/07_context_length.md)).
