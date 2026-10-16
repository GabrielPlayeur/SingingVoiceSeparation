# Ideas not worked on yet

These came out of the first training and my reading. They're further away, mostly because of the 4 GB budget or because they change several things at once.

- **Deal with the phase.** The mixture phase limits the model wherever the accompaniment dominates. Options: give the network the complex spectrogram, or estimate a complex mask whose magnitude can exceed 1, as in [Kong et al. (ISMIR 2021)](https://arxiv.org/abs/2109.05418), which decouples magnitude and phase estimation.
- **Better intermediate blocks**, like the TFC-TDF blocks of [Choi et al. (ISMIR 2020)](https://arxiv.org/abs/1912.02591).
- **Attention at the bottleneck.** After 6 stride-2 layers, each bottleneck position covers about 0.75 s × 1.4 kHz, so attention there would only see very coarse tokens. [07](07_context_length.md) showed that a longer context doesn't help this network, so attention would have to help some other way. I'd need to try it to know.
- **A second stage** that predicts a residual to clean up what the mask missed, as in [TS-BSMamba2 (Bai et al., 2024)](https://arxiv.org/abs/2409.06245).
- **Masks for several sources with shared weights**, as in [Kadandale et al. (2020)](https://arxiv.org/abs/2003.10414), so the instruments "compete" with the voice for each bin.
- **A generative approach**, which would recreate a clean voice in bins shared with instruments instead of averaging them. Probably below state of the art, but a different trade-off.
