# Protocol notes

Moved from the README. They describe the 150-epoch study; the re-run at 1000 epochs is in [09](09_convergence_probe.md), [10](10_recipe.md) and [11](11_final.md).

## Where training runs

Every model is sized for my 4 GB GPU (GTX 1650). The baseline peaks at 1.37 GiB in training. To go faster, I also rented an A800 (AutoDL) and trained several runs in parallel there. The 4 GB rule is about the model, not the machine, so I measure the peak VRAM locally before adding a model.

Trained locally: the baseline, the mono model, the cosine run and the int8 experiment. Trained on the server: the other learning-rate runs, experiments 04 to 07, and the whole re-run. Same code, data, split and seed.

## Noise between runs

The baseline was trained three times with identical code, config and seed (once locally, twice on the server). Median test SDR: 4.50, 4.47 and 4.78 dB. One run's median moves by up to 0.3 dB, but the **paired** per-track differences stay within about ±0.15 dB. That was the yardstick of the 150-epoch pages. It only measured GPU non-determinism; the seed-to-seed spread was measured later: σ_seed = 0.089 dB on validation ([10](10_recipe.md)).

## The baseline and the head bug

The model follows Jansson et al. 2017, with the changes listed in [01](01_baseline_vs_oracle.md#differences-from-the-paper):

- Input: magnitude STFT of the mixture (44.1 kHz, window 2048, hop 512), one channel per audio channel.
- U-Net: 6 conv encoder blocks (5×5, stride 2, 16 → 512 channels), 6 transposed-conv decoder blocks with skip connections, 50 % dropout on the first 3 decoders. 9.8 M parameters.
- Output: a mask in [0, 1] (sigmoid). Vocals = mask × mixture magnitude, with **the mixture's phase**, then inverse STFT.
- Training: L1 on magnitudes, Adam 1e-3, random 5.93 s crops, 150 epochs of 500 crops, batch 8.

**The head bug.** My first version kept BatchNorm + ReLU before the final 1×1 conv and sigmoid. The ReLU output is ≥ 0, so the mask was capped at `sigmoid(bias)` whenever the head weights were negative: in that checkpoint it never went above 0.727, and sat exactly there on 46 % of the bins. Fixed in `68ba08f`. The controlled rerun at 1000 epochs found no measurable cost (−0.01 dB, [11](11_final.md)): that run learned positive weights, so its mask wasn't capped.
