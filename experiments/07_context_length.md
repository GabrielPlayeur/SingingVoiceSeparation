# 07 · Context length (Planned)

[← back to README](../README.md)

> **Planned, not run yet.** The receptive field, the VRAM figures and everything I quote from the paper are measured or read; every statement about quality is a prediction.

**Why.** The baseline sees random crops of 5.93 s (261,632 samples, 512 STFT frames). Nothing in the project says why that length: it came with the first implementation. [Rouard et al. 2022](https://arxiv.org/abs/2211.08553) (HT Demucs) report that training on longer excerpts helps a lot, so it's worth testing here. It's also the control for the "attention at the bottleneck" idea in the README: if a longer context doesn't help this convolutional network, I learn that context isn't what limits it before spending effort on attention.

## What the paper says (and what it doesn't)

Read in the arXiv PDF (arXiv:2211.08553; the page gives no venue, to check before citing). Table 2, "Impact of segment duration, transformer depth and transformer dimension", SDR over all four sources on MUSDB18-HQ + 800 extra songs:

| segment | depth | dim | parameters | SDR (All) |
|---|---|---|---|---|
| 3.4 s | 5 | 384 | 26.9 M | 8.17 |
| 3.4 s | 7 | 384 | 34.0 M | 8.26 |
| 3.4 s | 5 | 512 | 41.4 M | 8.12 |
| **7.8 s** | 5 | 384 | 26.9 M | **8.70** |
| 7.8 s | 7 | 384 | 34.0 M | out of memory |
| 7.8 s | 5 | 512 | 41.4 M | 8.80 |
| 12.2 s | 5 | 384 | 26.9 M | out of memory |

- **3.4 s → 7.8 s: +0.53 dB** (8.17 → 8.70; the text says "almost 0.6 dB").
- **The "+0.4 dB more at 12.2 s" is not in Table 2.** A dense 12.2 s model runs out of memory there. The 12.2 s result comes from § 5.5, with **sparse attention kernels, depth 7 and dim 512 all changed at once**: 8.94 dB, i.e. **+0.14 dB** over the 7.8 s / dim 512 row (8.80). The conclusion speaks of "a supplementary gain of 0.4 dB" from the longer input, which doesn't match § 5.5's 0.14 dB; I only trust the number I can trace to a table, and it's confounded. So the sourced effect is **+0.5 dB from doubling the segment, then at most +0.14 dB, mixed with other changes**.
- Their context is a **cross-domain Transformer encoder**: self-attention over the whole segment, so a longer segment gives every position a longer context by construction. Training (§ 5.1): 8 V100 GPUs (32 GB), batch 32, 1,200 epochs of 800 batches, L1 loss on the waveform, Adam 3e-4, exponential moving average, repitching and remixing augmentation, on MUSDB18-HQ plus 800 curated songs from an internal dataset. At test time chunks have the training duration with 25% overlap.

Neither condition holds here: my network is a plain U-Net with no attention, one source, 84 training tracks, a 4 GB card. Their numbers are a motivation, not a target.

## Why it may not transfer: the receptive field (measured)

`uv run python -m analysis.receptive_field`: I back-propagate the mask of one central output frame to the input and see which input frames get a non-zero gradient (random weights, eval mode, 2,048-frame input).

- **Total receptive field: 381 frames = 4.42 s**, smaller than the 5.93 s crop. No output frame of this U-Net depends on anything further than ±2.2 s, so **a longer crop doesn't give any frame more context**.
- **Effective receptive field: 45 frames = 0.52 s** (gradient above 1e-3 of the maximum; arbitrary threshold, untrained weights, so an order of magnitude only).

What a longer crop does change here is the **edge effect**: frames within half a receptive field of a crop edge see zero padding instead of music. Share of output frames affected:

| crop | frames | with the total field (4.42 s) | with the effective field (0.52 s) |
|---|---|---|---|
| 2.96 s | 256 | 100% | 17% |
| 5.93 s (baseline) | 512 | 74% | 9% |
| 8.90 s | 768 | 49% | 6% |
| 11.88 s | 1024 | 37% | 4% |

So at 2.96 s every frame is edge-affected, and the gains from 5.93 s to 11.88 s are limited to fewer padded frames. At inference, the 50% overlap with Hann weighting already down-weights the edges (`separate_full_track`), which should make the training-time edge effect matter even less. This is the reason I expect much less than HT Demucs' +0.5 dB.

## Hypotheses and predictions

All estimates, not measurements.

- **H1, training length.** A model trained on longer crops separates better, with a saturation once the crop is well beyond the receptive field. Prediction: 2.96 s clearly worse than the baseline (every frame is edge-affected; −0.2 to −0.5 dB), 8.90 s and 11.88 s at most +0.1 to +0.2 dB. **If rejected** (no gain beyond the noise): the context isn't a limit for this convolutional model, and attention at the bottleneck can't be justified by "longer context" alone. If the gain is **≥ 0.3 dB**, something beyond the edge effect is at work (see the control).
- **H2, what exactly changes.** With the batch fixed at 8, a longer crop also means **more audio per step** (×2 at 11.88 s, so ≈ 247 h drawn over 150 epochs instead of 124 h, ≈ 46 passes instead of ≈ 23 over the 5.3 h of training audio). The control `ctx12_b4` (batch 4) keeps the audio per step and the number of steps of the baseline:
  - `ctx12 ≈ ctx12_b4 > baseline` → the context length helps;
  - `ctx12 > ctx12_b4 ≈ baseline` → what helps is the amount of audio per step (effective batch), not the context;
  - `ctx12 ≈ ctx12_b4 ≈ baseline` → no effect.
- **H3, inference length alone.** Evaluating the *baseline checkpoint* with other window lengths, without retraining: expected within ±0.05 dB, since the network is fully convolutional and the receptive field is 4.4 s. A bigger change would mean the BatchNorm statistics or the padding matter.

## Setup

Only `data.crop_length` changes (plus batch size and epoch size for the control). Same STFT, depth 6, loss, LR 1e-3 constant, seed 42, 150 epochs, early stopping patience 30.

| config | crop (samples) | seconds | STFT frames | batch | crops/epoch | steps/epoch | peak VRAM (measured) |
|---|---|---|---|---|---|---|---|
| ctx3 | 130,560 | 2.96 | 256 | 8 | 500 | 63 | 0.70 GiB |
| **baseline** (`stereo`, from `loss-v1`) | 261,632 | 5.93 | 512 | 8 | 500 | 63 | 1.28 GiB |
| ctx9 | 392,704 | 8.90 | 768 | 8 | 500 | 63 | 1.87 GiB |
| ctx12 | 523,776 | 11.88 | 1024 | 8 | 500 | 63 | 2.44 GiB |
| ctx12_b4 (control) | 523,776 | 11.88 | 1024 | 4 | 252 | 63 | 1.28 GiB |

VRAM: GTX 1650 4 GB, Adam, fp32, random input, 6 steps, no data loading, so a bit below a real training (1.28 vs 1.37 GiB for the baseline, [04](04_waveform_loss.md)). All fit in 4 GB. I don't report step times: they were disturbed by other jobs on the GPU during the measurement.

Why these lengths. The crop must give a number of frames that is a multiple of `2**depth` = 64 (`assert_no_temporal_padding`), so 256, 512, 768, 1024 frames. That is 0.5×, 1×, 1.5× and 2× the baseline, which brackets the 3.4 → 7.8 → 12.2 s of the paper (×2.3 then ×1.6) at a comparable ratio. **I don't go to 18 or 24 s**: four training tracks are shorter than 20 s (12.9, 17.3, 17.3 and 19.6 s) and `MUSDBTrainDataset` draws a track with a probability proportional to `length − crop`, which becomes negative. Even at 11.88 s the 12.9 s track gets a weight of 0.006% (measured), i.e. it's practically never drawn, and the sampling weights tilt a little further towards long tracks. A longer crop would need a code change (drop or pad short tracks), which would be another variable.

The baseline point is the `baseline_s42` run of `loss-v1` (same code, same seed, same machine), not retrained.

## Evaluation protocol

Three sets of numbers, all on the 50 test tracks, full-track museval SDR, paired against the baseline:

1. **Same inference windows for everyone (main result).** `run_queue.py` evaluates every run with the default 5.93 s windows (`src.evaluate`, 50% overlap). Differences come from training only.
2. **Each model at its own length.** `deploy/eval_crops.py --eval ctx3_s42=130560 …` re-evaluates the checkpoint with windows of its training length (`src.evaluate --crop-length`). This is the HT Demucs protocol (chunks = training duration).
3. **The baseline checkpoint at other lengths** (H3), same script, no retraining.

Decision metric: the paired median SDR difference against the baseline (95% bootstrap CI, Wilcoxon), as in the other experiments, plus the slope in dB per doubling of the crop. **I don't use the validation SI-SDR across runs**: `MUSDBDataModule` draws the 15 validation crops per track with the same `crop_length`, so each run validates on different excerpts, and the checkpoint selection uses them.

## Known limits, stated before running

- **One seed.** The baseline differs by about 0.1 dB between two trainings ([README](../README.md)); the 3-seed runs (`seeds` group) will say how much. A gap below ~0.15 dB isn't interpretable.
- **Fixed number of steps, not of audio.** ctx9 and ctx12 see more audio than the baseline; ctx3 sees less (the reverse). Only `ctx12_b4` removes this confound, and only at 11.88 s.
- **The receptive field was measured on a width-16, depth-6 network, with untrained weights** for the effective value. Rerun with `--ckpt` on a trained checkpoint to check the 0.52 s.
- **Batch size and BatchNorm.** The control uses batch 4 on larger maps; BatchNorm statistics are computed over (batch, time, frequency), so they should barely change, but I haven't checked.
- **Not a replication.** HT Demucs is a hybrid waveform/spectrogram transformer with four sources, much more data and a different SDR average; I only borrow the question.

## Running it on the server

`context` group of [deploy/queue.yaml](../deploy/queue.yaml) (ctx3, ctx9, ctx12, ctx12_b4), seed 42, tag `ctx-v1`. Step time grows roughly with the crop length, so ctx12 should take about twice the baseline's (⚠ estimate, not measured on the server; the baseline took ≈ 17 min with 3 runs in parallel in `lr-v4`).

```bash
git tag ctx-v1 && git push origin feat/stereo ctx-v1
# server
bash deploy/setup_server.sh ctx-v1
.venv/bin/python deploy/run_queue.py --tag ctx-v1 --group context --parallel 4 --dry-run
.venv/bin/python deploy/run_queue.py --tag ctx-v1 --group context --parallel 4
# after DONE: evaluations 2 and 3 (about 2.5 min each; the baseline checkpoint comes from the loss-v1 run)
.venv/bin/python deploy/eval_crops.py --tag ctx-v1 --parallel 2 \
    --eval ctx3_s42=130560 --eval ctx9_s42=392704 --eval ctx12_s42=523776 --eval ctx12_b4_s42=523776 \
    --eval loss-v1/baseline_s42=130560,392704,523776
# PC
deploy/fetch_results.sh ctx-v1 --wait
uv run python -m analysis.context_length \
    --train 2.96=outputs/server/ctx-v1/ctx3_s42/eval.json --train 5.93=outputs/server/loss-v1/baseline_s42/eval.json \
    --train 8.90=outputs/server/ctx-v1/ctx9_s42/eval.json --train 11.88=outputs/server/ctx-v1/ctx12_s42/eval.json \
    --control 11.88=outputs/server/ctx-v1/ctx12_b4_s42/eval.json \
    --own 2.96=outputs/server/ctx-v1/ctx3_s42/eval_crop130560.json --own 8.90=outputs/server/ctx-v1/ctx9_s42/eval_crop392704.json \
    --own 11.88=outputs/server/ctx-v1/ctx12_s42/eval_crop523776.json \
    --infer 2.96=outputs/server/ctx-v1/infer_loss-v1_baseline_s42/eval_crop130560.json \
    --infer 8.90=outputs/server/ctx-v1/infer_loss-v1_baseline_s42/eval_crop392704.json \
    --infer 11.88=outputs/server/ctx-v1/infer_loss-v1_baseline_s42/eval_crop523776.json
```

Don't pass `--shutdown` to the queue if the `eval_crops.py` step still has to run: the instance would stop once `FETCHED` is set. Output: `figures/context_length.png` and `.json`.

**Result:** not run yet.

## References

- S. Rouard, F. Massa, A. Défossez, *Hybrid Transformers for Music Source Separation*, [arXiv:2211.08553](https://arxiv.org/abs/2211.08553), 2022. § 5.1 to § 5.5 and Table 2 read in the PDF text; the arXiv page gives no venue (ICASSP 2023 from memory, not verified). I haven't read the architecture sections (§ 3 and § 4) closely.
