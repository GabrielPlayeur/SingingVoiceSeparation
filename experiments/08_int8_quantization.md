# 08 · int8 quantization (post-training and QAT)

[← back to README](../README.md)

**Status.** Done, run locally (GTX 1650 for the simulation and QAT, CPU for the real int8) on one checkpoint: the stereo baseline, one seed. Validation numbers are measured on fixed excerpts; test numbers are on the 50 test tracks. Nothing here is a statement about other models: the same pipeline works on any checkpoint (`python -m src.quantize`), but I only ran it on this one.

**Result in one paragraph.** Weights in int8 are free (−0.03 dB); the cost is in the activations, and almost all of it comes from the first two encoder layers with min-max ranges. Clipping the activation ranges at the 99.99th percentile brings post-training quantization to **−0.02 dB** on the test set (not distinguishable from fp32); QAT brings the real int8 model to **4.49 dB vs 4.50 dB** (−0.00 dB [−0.07, +0.07]). The model is **4× smaller (39.3 → 9.9 MB)**, but on this CPU int8 is **2.3× slower than fp32**, because of the transposed convolutions.

**Question.** The project asks what a small model can do under a 4 GB constraint. Quantizing the weights to int8 should divide the model size by 4. How much quality does it cost, can quantization-aware training (QAT) get it back, and does int8 actually run faster on my CPU?

## What I read

- **Nagel, Fournarakis, Amjad, Bondarenko, van Baalen, Blankevoort, *A White Paper on Neural Network Quantization*, [arXiv:2106.08295](https://arxiv.org/abs/2106.08295), 2021.** Sections 2 to 4 read in the PDF text. What I used, with section numbers:
  - **Simulation** (§ 2.3): quantization can be simulated with fake quantizers in floating point, so the accuracy of any setting can be tested on a GPU.
  - **BatchNorm folding** (§ 2.3.1) and static folding for QAT, which is "very effective" when starting from a converged model (§ 4.2, Table 7).
  - **Layers without hardware support** (§ 2.3.2): for an activation that can't be fused (here LeakyReLU, and the final sigmoid), a quantizer goes before and after it. **Concatenation** (§ 2.3.3): the branches have different grids, so a requantization step is needed.
  - **Symmetric weights, asymmetric activations, per-channel weights** (§ 2.4.1, § 2.4.2).
  - **Range setting** (§ 3.1): min-max is sensitive to outliers; MSE range setting clips them. Table 1 and Table 2 compare them at 8, 6 and 4 bits.
  - **Debugging flow** (§ 3.7, Figure 9): check that the quantized model with quantizers off equals fp32; quantize weights and activations separately; then one quantizer at a time. I followed this order.
  - **QAT** (§ 4): straight-through estimator (§ 4.1); initialize from a PTQ model (§ 4.3: the benefit of a better initialization "vanishes the longer training lasts"); recommended pipeline (§ 4.4) with **learnable** quantization ranges; Adam doesn't need a separate learning rate for them.
- **Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference: A whitepaper*, [arXiv:1806.08342](https://arxiv.org/abs/1806.08342), 2018.** Sections 2 to 7 read. Used for: per-channel weights and per-layer activations as the preferred scheme (abstract, § 2.6); weight-only int8 divides the model size by 4 even without int8 arithmetic (abstract, § 3.1.1); about 100 calibration mini-batches (§ 3.1.2); BatchNorm handling in QAT (§ 3.2.2, § 4); fine-tune from a floating point checkpoint (§ 4); "an over-parameterized model is more amenable to quantization" (§ 5); and the remark that 4 and 8 bits are enough for classification but "higher precision support is likely needed for regression applications" (§ 7). A mask is a regression output.

Both papers evaluate image classification, segmentation and detection (and BERT for Nagel); neither covers audio or masks. They are guides to the method, not predictions for this model.

## Setup

**Model.** The stereo baseline U-Net (9.82 M parameters, [01](01_baseline_vs_oracle.md)). The STFT, the mask application and the inverse STFT stay in floating point; only the U-Net is quantized. Its input is |X| (the baseline has no input transform).

**Two code paths with the same quantizer positions** ([src/quantization.py](../src/quantization.py), CLI in [src/quantize.py](../src/quantize.py)):

- **Simulation** (`SimUNet`): fake quantizers in floating point (PyTorch's `fake_quantize_*`, straight-through gradient), any bit width, GPU. Used for the ablations, the per-layer analysis and QAT.
- **Real int8** (`export_int8`): PyTorch's eager `torch.ao.quantization` with the x86 (fbgemm) kernels, CPU only. Used for size and latency. ⚠ This API is flagged as deprecated in PyTorch 2.10 (migration to torchao); it still works and the version is pinned by `uv.lock`.

The two agree closely but not always exactly (48 validation excerpts, 3 per track):

| setting | val SI-SDR, simulation | val SI-SDR, real int8 | mean absolute mask difference |
|---|---|---|---|
| PTQ min-max, 7-bit activations | 3.230 | 3.236 | 0.00014 |
| PTQ min-max, 8-bit activations | 3.759 | 3.753 | 0.00000 |
| PTQ percentile 99.99, 7 bits | 4.035 | 3.966 | 0.0029 |
| PTQ percentile 99.99, 8 bits | 4.112 | 4.143 | 0.0049 |
| QAT (LR 3e-5), 7 bits | 4.433 | 4.403 | 0.0020 |

With min-max ranges they are identical; with clipped ranges the real kernels differ by up to 0.07 dB (⚠ I didn't find the cause; the first suspect is how the real kernels round when many values saturate). The ablations below are simulations; the final numbers also include the real int8 model.

**Where the quantizers are** (25 activation quantizers, 13 weight quantizers):

| tensor | quantizer |
|---|---|
| network input | `in` |
| each encoder, 6 | after the convolution (`enc{i}.conv`) **and** after LeakyReLU (`enc{i}.act`) |
| each decoder, 6 | after the transposed convolution (`dec{i}.conv`); ReLU doesn't change the grid |
| each skip concatenation, 5 | `cat{i}`: requantization to a shared grid |
| final 1×1 convolution | `final.conv`; the sigmoid stays in floating point |

**Choices and where they come from.**

| choice | value | reason / source |
|---|---|---|
| BatchNorm | folded statically into the (transposed) convolution, exact to ~1e-5 on the output | Nagel § 2.3.1, § 4.2 |
| weights | symmetric int8; per-channel for the encoder convolutions and the 1×1; **per-tensor for the transposed convolutions** | per-channel for transposed convolutions isn't supported by `torch.ao` ("Per channel weight observer is not supported yet for ConvTranspose{n}d"); the simulation can also do them per-channel to measure what that costs |
| activations | asymmetric (affine), per-tensor, **7 bits** (`reduce_range`, 0..127) | Nagel § 2.4.1; 7 bits is PyTorch's recommendation for CPUs without VNNI (mine has AVX2 only), because the int8 kernels can saturate an int16 accumulator (⚠ from PyTorch's documentation of `reduce_range`, as I remember it, not checked). **I saw no saturation on this CPU**: with full 8-bit activations the real kernels match the simulation exactly (table above), and 8 bits is better with min-max ranges (see below); so 7 bits was a conservative choice that all the results below inherit |
| ranges | min-max, a percentile (99.99 or 99.9), or an MSE grid search, from 64 random **training** crops (fixed seed) | Nagel § 3.1; Krishnamoorthi § 3.1.2 uses ~100 mini-batches; Nagel's footnote 4 says 500 to 1,000 images for classification |
| not done | cross-layer equalization, bias correction, AdaRound, learnable ranges | CLE matters mostly for depthwise-separable networks (Nagel § 3.2), which this isn't; the others are the next step if the PTQ gap needs closing |

**Metrics.**
- **Fast, for the ablations:** median SI-SDR over **240 fixed validation crops** (16 tracks × 15 crops of 5.93 s). It is the quantity the training monitors (`val/si_sdr_median`), so the fp32 reference is 4.136 dB. These are not the test numbers and are not museval SDR.
- **Final, for the decision:** museval SDR median over the 50 test tracks, paired against the fp32 model, as in the other experiments (`python -m src.quantize evaluate`).
- The baseline's validation SI-SDR is noisy between epochs (std of the difference between two consecutive epochs: 0.32 dB at constant LR, [03](03_lr_schedule.md)). That noise is about the training, not about this measurement (fixed crops, a fixed model), but it explains why the QAT runs below use a cosine learning rate and I keep the **last** epoch rather than the best one, which would be picked on noise.

**Predictions** (written after a first prototype of the real int8 model, before the ablation grid). (1) int8 weights alone should cost almost nothing (Krishnamoorthi § 3.1.1, Nagel Table 1), 4 bits should hurt. (2) Quantizing the activations with min-max would hurt, and I expected the **input** to be the culprit: |X| spans 58 dB between its median (0.059) and its 99.9th percentile (47), see [05](05_amplitude_compression.md). (3) QAT should recover most of the gap.

## Results

### 1. Sanity check and weights vs activations

All values are the change in median validation SI-SDR against fp32 (4.136 dB). `quantizers off` gives exactly fp32 (+0.000), the first step of Nagel's flow (§ 3.7).

![PTQ ablations and per-quantizer sensitivity](../figures/quant_ptq.png)

| setting (PTQ, 64 calibration crops, simulation) | Δ val SI-SDR |
|---|---|
| weights only, int8, per-channel convolutions | −0.03 dB |
| weights only, int8, per-tensor everywhere | −0.00 dB |
| weights only, int8, transposed convolutions per-channel too | −0.02 dB |
| weights only, int8, MSE range | −0.01 dB |
| weights only, 6 bits | −0.04 dB |
| weights only, **4 bits** | **−1.00 dB** |
| activations only, 7 bits, min-max | **−1.15 dB** |
| activations only, 7 bits, 99.99th percentile | −0.14 dB |
| activations only, 7 bits, 99.9th percentile | −0.24 dB |
| activations only, 7 bits, MSE | −0.52 dB |
| activations only, 8 bits, min-max | −0.21 dB |
| activations only, 8 bits, 99.99th percentile | −0.16 dB |
| activations only, 7 bits, min-max, input kept in floating point | −1.12 dB |
| **W8A8, 7 bits, min-max** (the default) | **−1.15 dB** |
| W8A8, 8 bits, min-max | −0.23 dB |
| **W8A8, 7 bits, 99.99th percentile** | **−0.16 dB** |
| W8A8, 8 bits, 99.99th percentile | −0.19 dB |
| W8A8, 7 bits, MSE (weights and activations) | −0.48 dB |
| W8A8, 7 bits, 99.99th percentile, input in floating point | −0.27 dB |

- **Weights are almost free**: int8 costs 0.0 to 0.03 dB, six bits is the same, four bits breaks. This matches Krishnamoorthi (§ 3.1.1) and Nagel (Table 1: W8 within 0.1 point of fp32 on ResNet18, 0.6 on MobileNetV2 with per-tensor min-max). Per-channel or per-tensor makes no visible difference at 8 bits, **including for the transposed convolutions** (−0.00 vs −0.02), so their per-tensor limitation in `torch.ao` doesn't cost quality here.
- **The activations are the whole problem**, and **not because of the input**: leaving the input in floating point barely changes anything (−1.12 vs −1.15 dB). My prediction (2) was wrong on the culprit.
- **Range setting is the fix**: clipping at the 99.99th percentile recovers almost everything (−1.15 → −0.16 dB). **MSE is better than min-max but worse than the percentile here** (−0.52 vs −1.15 and −0.14 for activations only; −0.48 for W8A8). In Nagel's Table 2 (ResNet18, classification) MSE and min-max are on par at 8 bits (69.59 vs 69.60) and MSE wins at 6 and 4 bits, so I expected it to do at least as well as the percentile; I did not investigate why (⚠ the MSE objective weights all values equally, which may over-clip the tails that carry the loud bins).
- **One bit matters with min-max, not with the percentile**: min-max with 8 full bits is −0.23 dB, against −1.15 dB with 7 bits (the `reduce_range` setting I chose as a precaution). Once the ranges are clipped, the extra bit changes nothing (W8A8 −0.19 dB at 8 bits vs −0.16 dB at 7). So the 7-bit precaution is harmless with percentile ranges and costly without them.
- **Reproducibility.** The percentile and MSE ranges are estimated on a random subsample of the activations. My first, unseeded run of the grid gave −0.32 dB for activations-only MSE and −0.43 for W8A8 MSE; the seeded rerun above gives −0.52 and −0.48. The MSE numbers move by up to 0.2 dB with the subsample, the percentile ones by about 0.03 dB: read the MSE rows as "unstable". Keeping the input in floating point does not help (−0.27 vs −0.16 dB with percentile ranges): the input quantizer is not the problem.

### 2. Which layers hurt

Nagel's per-layer analysis (§ 3.7): only one quantizer is active, all the others are in floating point (120 validation crops, fp32 4.271 dB, W8A8 min-max):

| quantizer | Δ val SI-SDR |
|---|---|
| `enc0.conv` | **−0.94 dB** |
| `enc1.conv` | **−0.60 dB** |
| `in` | −0.26 dB |
| `enc0.act` | −0.24 dB |
| `enc1.act` | −0.17 dB |
| `enc4.act`, `enc2.act` | −0.11, −0.10 dB |
| `dec4.conv`, `cat4` | −0.07, −0.07 dB |
| **weights**: `enc0` | −0.10 dB |
| every other quantizer (the 6th encoder, all decoders, all other weights) | within ±0.04 dB |

The damage sits in the **first two encoder layers**, the ones that work at full resolution on the raw spectrogram. Why min-max fails there, from the observed ranges (`python -m src.quantize ranges`): for `enc0.conv` the observed range is [−110.6, 87.3] while the 0.01st and 99.99th percentiles are [−24.3, 16.4]; the min-max step is 1.56 against 0.32 with the percentile (about 5×), and the maximum is 5.3× the 99.99th percentile. Every quantizer has a maximum 2.6 to 6.9× its 99.99th percentile, so outliers are everywhere; what singles out the first layers is the **absolute step**: with min-max it is 1.56 (`enc0.conv`), 1.08 (`enc1.conv`), 0.86 (`enc0.act`) and 2.12 for the input, against 0.2 to 0.6 for most deeper layers. But step size alone doesn't explain the ranking: `dec5.conv` has a step of 1.08 and costs only 0.01 dB, so how a layer's error propagates also matters (⚠ I haven't separated the two). The sensitivity numbers are not additive either: summed they exceed the combined −1.15 dB.

### 3. QAT

Starting from the PTQ model (99.99th percentile ranges, frozen during training), BatchNorm folded, weights re-quantized at every step per channel, the straight-through estimator. 10 epochs of 500 crops (like the baseline's epochs, ~25 s each on the GTX 1650), Adam, cosine learning rate to 0, **the last epoch kept**. I tried four learning rates; the baseline was trained at 1e-3.

![QAT curves](../figures/quant_qat.png)

| QAT run (validation, 240 crops) | start (PTQ) | after 10 epochs | Δ vs fp32 (4.136) |
|---|---|---|---|
| percentile init, LR 1e-5 | 3.973 | 4.081 | −0.055 |
| percentile init, **LR 3e-5** | 3.973 | **4.114** | **−0.022** |
| percentile init, LR 1e-4 | 3.973 | 4.040 | −0.096 |
| percentile init, LR 3e-4 | 3.973 | 3.978 | −0.158 |
| min-max init, LR 1e-4 | 2.982 | 3.611 | −0.525 |
| **control: no quantization**, LR 3e-5 (fp32 fine-tuning, same 10 epochs) | 4.136 | 4.105 | −0.031 |

- QAT with a small learning rate brings the gap from −0.16 dB (PTQ) to **−0.02 dB** (3e-5), which is inside the epoch-to-epoch noise of the curves (±0.03 dB with this schedule): I can't call the remaining gap real. **The control matters**: fine-tuning the same model in fp32 with the same schedule also ends at −0.03 dB, so on validation the quantized model matches an equally fine-tuned fp32 one. Fine-tuning itself does not improve the validation score here. A larger learning rate is worse (3e-4 undoes the PTQ), as expected when fine-tuning a converged model.
- **The initialization matters a lot here**, unlike in Nagel's Table 8 (§ 4.3: "the gain is usually small and vanishes the longer the training lasts"): from min-max ranges QAT only gets to −0.53 dB. The difference with Nagel is that my activation ranges are **frozen** after calibration, while they recommend learnable ranges (§ 4.4), which could move a bad initial range. I haven't tried it, so I can't say the learnable ranges would close the gap.
- The four learning rates are compared on the same validation crops that I used to pick the best one, so the best of four is slightly optimistic; the test set (next section) is the honest number. One seed, one run per setting.

### 4. Test set

museval SDR over the 50 test tracks, same protocol as the other experiments. The fp32 reference is the baseline checkpoint itself (`outputs/eval_stereo.json`, 4.50 dB). Differences are paired per track: median of the per-track differences, 95% bootstrap CI.

| model (all int8 W8A8 with 7-bit activations unless noted) | SDR | SI-SDRi | Δ SDR vs fp32 (paired) | better on |
|---|---|---|---|---|
| fp32 baseline | 4.50 | 9.31 | | |
| PTQ, min-max ranges (simulation) | 3.72 | 8.46 | **−0.63** [−0.75, −0.48] | 4/50 |
| PTQ, 99.99th percentile ranges (simulation) | 4.40 | 9.36 | **−0.02** [−0.07, +0.01] | 21/50 |
| QAT, LR 3e-5 (simulation) | 4.57 | 9.40 | +0.05 [−0.00, +0.13] | 31/50 |
| **QAT, real int8 on CPU** | **4.49** | 9.44 | **−0.00** [−0.07, +0.07] | 24/50 |
| *control: fp32 fine-tuned with the same schedule, no quantization* | 4.70 | 9.36 | +0.08 [+0.03, +0.16] | 33/50 |

- **Post-training quantization with percentile ranges is not distinguishable from fp32** on the test set (−0.02 dB, CI includes 0). With min-max ranges it loses 0.63 dB (worse on 46 of 50 tracks), as predicted by the validation grid (−1.15 dB there).
- **QAT looks like a gain (+0.05 dB, p = 0.015 in a Wilcoxon test) but it isn't one.** The control, fp32 fine-tuned with the same schedule and no quantizer, gains **+0.08 dB** [+0.03, +0.16] on the test set: the extra low-learning-rate epochs help the baseline by themselves (the baseline's curve was still rising at epoch 150, [03](03_lr_schedule.md)). Against that control, QAT in simulation is −0.02 dB [−0.05, +0.03] and the **real int8 model is −0.07 dB [−0.12, −0.03]**. The honest reading: the quantization cost after QAT is about 0.0 to 0.1 dB, and the deployable model is as good as the original baseline, not better.
- **Simulation and real int8 differ by 0.07 dB on the test set** (4.57 vs 4.49; paired −0.07 [−0.11, −0.03]), more than on validation (0.03 dB). The real model is the one that counts. (⚠ cause not found, see the table in the setup.)
- The test-set PTQ numbers are simulations; I did not run the real kernels on the test set for the PTQ models (≈ 35 min each on this CPU).

### 5. Real int8: size and speed

| | fp32 | int8 |
|---|---|---|
| `state_dict` on disk | 39.34 MB (BatchNorm included) | **9.87 MB** (3.98× smaller) |
| latency, 1 thread (median of 7, one 5.93 s excerpt through the U-Net) | 251 to 258 ms | 585 to 611 ms (**0.42× the speed**) |
| latency, 4 threads | 116 to 131 ms | 266 to 270 ms (**0.44× the speed**) |

- **The model is 4× smaller, but int8 is 2.3× slower than fp32 on this CPU.** The other measurements I took, with other jobs running on the machine, are noisier but in the same range (int8 0.33× to 0.57× the speed).
- **The transposed convolutions are why.** A profile of the int8 model on 1 thread: `quantized::conv_transpose2d` takes 508 ms of 582 ms (87%, 6 calls), `quantized::conv2d` 58 ms (7 calls), the rest is under 10 ms. In fp32 the 13 convolutions take 182 ms of 198 ms. So the int8 `Conv2d` is fast, but the int8 `ConvTranspose2d` takes 508 ms in total, against at most 182 ms for all 13 convolutions in fp32: roughly 4 to 10 times slower than the fp32 transposed convolutions (⚠ an estimate from the totals; I didn't time the fp32 transposed convolutions separately). The `onednn` backend refuses the layers ("ONEDNN only supports zero output_padding") and `qnnpack` is slower (1.3 s).
- Krishnamoorthi reports 2× to 3× speed-ups on phone CPUs (abstract, § 6, ResNet and MobileNet models, no transposed convolutions). That is a different CPU and a different layer mix; my result doesn't contradict it.
- The quantization helps the **size** (and so the memory to load the model), not the speed, for this architecture on this CPU. Making int8 fast would mean changing the decoder (an upsample followed by a convolution, or a sub-pixel formulation), which is a different model and a different experiment.

## What this says

- **int8 is cheap in quality for this model if the activation ranges are clipped.** Weights in int8 cost nothing; the cost is in the activations of the first two encoder layers with min-max ranges; a percentile range fixes it for post-training quantization, and QAT is not needed to reach the baseline's score (PTQ −0.02 dB [−0.07, +0.01] in simulation). QAT helps against the validation gap (−0.16 → −0.02 dB) but the test set can't separate it from the effect of fine-tuning.
- **This is also an answer to the 4 GB question: the model is 4× smaller at the same quality.** For size/quality, int8 shifts each point of the [06](06_size_quality.md) curve left by a factor of 4 in bytes; Krishnamoorthi's remark that an over-parameterized model is more amenable to quantization (§ 5) predicts that w32 would quantize at least as well as w16. I haven't tested it: the checkpoints of the size runs are on the server (`deploy/fetch_results.sh size-v1 --ckpt w32_s42`, then the same commands).
- **The raw spectrogram's dynamic range is again the issue** (58 dB between the median and the 99.9th percentile, [05](05_amplitude_compression.md)), but through the first layers' activations, not through the input quantizer. A network trained on |X|^0.3 (`power_in`) might need less clipping; I haven't tested it (I don't have its checkpoint locally).
- **Mixed precision is the natural next step** (Nagel § 3.7, "fixing individual quantizers"): keep `enc0.conv` and `enc1.conv` in higher precision, since they cause almost all of the loss with min-max ranges. With percentile ranges the remaining gap is already inside the noise, so it may not be needed.

## Limits

- **One checkpoint, one seed.** The ±0.03 dB noise of the QAT curves and the ~0.1 dB seed-to-seed variability of the baseline ([README](../README.md)) bound what I can claim. Differences under ~0.1 dB between settings are not interpretable.
- **Calibration and QAT use training crops; the learning rate and the range method were chosen on validation crops.** The test set was not used to choose anything.
- **Only W8A8 has real kernels.** The 6-bit and 4-bit numbers are simulations (there is no int4 kernel here); they show where the cliff is, not a deployable model.
- **Test-set numbers for PTQ are simulations**; only the QAT model was run with the real kernels on the 50 tracks, and the real kernels differ from the simulation by up to 0.07 dB.
- **7-bit activations everywhere** (a precaution that turned out not to be needed on this CPU, see the setup); with percentile ranges it makes no difference, with min-max it costs about 0.9 dB.
- **Numbers are for this CPU and this software stack** (Intel i5-10300H, AVX2, no VNNI, PyTorch 2.10, x86 backend). On a CPU with VNNI, or a phone, the speed could be very different.
- Not tested: dynamic quantization, per-channel transposed convolutions in the real kernels (not available), learnable ranges, mixed precision, the other checkpoints (`power_in`, the size curve).

## Reproduce

```bash
CK=outputs/VoiceSeparation/fgvzjxua/checkpoints/baseline-stereo.ckpt
uv run python -m src.quantize sweep $CK -o outputs/quant/sweep_baseline.json          # ablation grid (validation)
uv run python -m src.quantize sensitivity $CK --max-batches 15 -o outputs/quant/sensitivity_baseline.json
uv run python -m src.quantize ranges $CK -o outputs/quant/ranges_baseline.json
uv run python -m src.quantize qat $CK --act-range percentile --epochs 10 --lr 3e-5 --cosine --tag p99_lr3e-5_cos \
    -o outputs/quant/baseline.qat_p99_lr3e-5_cos.pt
uv run python -m src.quantize evaluate $CK --qat outputs/quant/baseline.qat_p99_lr3e-5_cos.pt -w 2 -o outputs/quant/test_qat.json   # simulation, GPU
uv run python -m src.quantize evaluate $CK --qat outputs/quant/baseline.qat_p99_lr3e-5_cos.pt --real -w 2 -o outputs/quant/test_qat_real.json   # real int8, CPU
uv run python -m src.quantize export $CK --qat outputs/quant/baseline.qat_p99_lr3e-5_cos.pt --cpu       # size, latency, simulation vs real
uv run python -m src.quantize qat $CK --no-weights --no-acts --epochs 10 --lr 3e-5 --cosine --tag fp32_ctrl \
    -o outputs/quant/baseline.qat_fp32_ctrl.pt      # control: same fine-tuning, no quantization (evaluate it with --qat)
uv run python -m analysis.quantization --sweep outputs/quant/sweep_baseline.json --sensitivity outputs/quant/sensitivity_baseline.json \
    --qat outputs/quant/baseline.qat_p99_lr3e-5_cos.pt
```

The machine has 8 GB of RAM: with another heavy job running, the QAT can be killed by the out-of-memory killer (it happened to one of my runs, which I reran). The training workers use `spawn` to avoid copying the validation crops.

## References

- M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, T. Blankevoort, *A White Paper on Neural Network Quantization*, [arXiv:2106.08295](https://arxiv.org/abs/2106.08295), 2021. Sections 2 to 4 read; the experiments (§ 3.6, § 4.5) are on image and text models.
- R. Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference: A whitepaper*, [arXiv:1806.08342](https://arxiv.org/abs/1806.08342), 2018. Read, except the appendix on BatchNorm.
