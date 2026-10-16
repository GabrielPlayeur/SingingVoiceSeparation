# Protocol notes

Moved from the README. Numbers are those of the 150-epoch study (1 seed per run) and are kept as the "short budget" column of the re-run.

## Where training runs

All models are sized for 4 GB. The baseline peaks at 1.37 GiB during training (measured on the GTX 1650). But to get through experiments faster, I also rent a GPU (an A800 80 GB on AutoDL) and train several runs in parallel there. The 4 GB rule is about the model, not the machine: server runs aren't capped, so it's on me to measure peak VRAM locally before adding a model. 

Trained locally: the baseline, the mono model, the `cosine` run, and everything in the int8 experiment. 

Trained on the server: the other learning-rate runs and all of experiments 04 to 07 (each group with its own retrained baseline). Same code, data, split and seed. I trained `lr3e4_cosine` on both machines with the same seed (0.05 dB apart on validation), and the baseline three times with identical code, config and seed (once locally, twice on the server). Their median test SDRs are 4.50, 4.47 and 4.78 dB: one run's median moves by up to 0.3 dB. But their **paired** per-track differences stay within about ±0.15 dB, so that's the yardstick I use for every comparison. It only measures GPU non-determinism, not seed variability, until the 3-seed runs are done.


## The baseline and the head bug

The model follows Jansson et al. 2017, with a few changes listed in [experiments/01](01_baseline_vs_oracle.md#differences-from-the-paper):

- Input: magnitude STFT of the mixture (44.1 kHz, 2048-point window, hop 512), one channel per audio channel.
- U-Net: 6 conv encoder blocks (5×5, stride 2, 16 -> 512 channels), 6 transposed-conv decoder blocks with skip connections, 50% dropout on the first 3 decoders. 9.8 M parameters.
- Output: a mask in [0, 1] (sigmoid). Estimated vocals = mask × mixture magnitude, recombined with **the mixture's phase**, then inverse STFT.
- Training: L1 loss on magnitudes, Adam 1e-3, random ~5.9 s crops, 150 epochs of 500 crops, batch 8.

**A bug I found along the way.** In my first version, the last decoder block still had BatchNorm + ReLU before the final 1×1 conv and sigmoid. Since the ReLU output is ≥ 0, the mask was capped at `sigmoid(bias)` and stuck at exactly that value wherever the ReLU output was zero. In the checkpoint trained with this bug, the mask never went above 0.727 and sat exactly at 0.727 on 46% of time-frequency bins (measured). I removed the ReLU/BN from that block (commit `68ba08f`). I don't have a clean before/after score: the only pre-fix run with a test score used an older evaluation pipeline and stopped at 47 epochs, so comparing it to the current baseline would mix several changes. A controlled rerun with the old head is on the to-do list.
