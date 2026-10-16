# 00 · The dataset: MUSDB18

[← back to README](../README.md)

I use **compressed MUSDB18** (AAC stems), not MUSDB18-HQ: 150 tracks, 100 for train/val and 50 for test. My validation set is 16 of the 100 train tracks, split by artist (`GroupShuffleSplit`, `val_size=0.14`, `random_seed=42`, saved in `musdb18/split.json`), so no artist appears in both train and val.

## Official mixture vs vocals + accompaniment

Because the stems are compressed separately, the official `mixture` isn't exactly `vocals + accompaniment`. I train on the sum (so the target is consistent with the input) and I wanted to know if that matters.

`uv run python -m analysis.mixture_gap` (150 tracks)

![mixture gap](../figures/mixture_gap.png)

- The residual `mixture − (vocals + accompaniment)` is **−30.7 dB** below the mixture (median; 10th percentile −28.4 dB; between −26 and −39 dB for every track except one, see below). So the difference is small.
- It's concentrated **above 17 kHz**, where the residual is louder than the mixture. My guess is the AAC encoder's low-pass, but I haven't checked. There is almost no useful energy there anyway.
- To see if it affects separation, I computed the oracle mask on the summed mixture and applied it to the real mixture (48 test tracks with an audible voice, 3 crops of 6 s each): median SI-SDR **10.21 dB** on the sum vs **10.17 dB** on the real mixture, a **0.05 dB** median difference. It's negligible for this problem.
- Exceptions: *PR - Oh No* (mixture only 7.6 dB above the residual) and *PR - Happy Daze*, two Native Instruments tracks where the voice is almost inaudible (median RMS −86 dB for *Happy Daze*, 67% of 1 s windows below −60 dB). There, the residual is louder than the voice (voice/residual SNR down to −9.4 dB).

Not done yet: actually training and evaluating on the official `mixture.wav`. The oracle test above suggests the effect is close to zero, but it's a proxy.

## Genres and artists

`uv run python -m analysis.dataset_distribution`

![distribution](../figures/dataset_distribution.png)

| Genre | train (84) | val (16) | test (50) |
|---|---|---|---|
| Pop/Rock | 33% | 44% | **72%** |
| Rock | 18% | 12% | 0% |
| Singer/Songwriter | 14% | 12% | 0% |
| Heavy Metal | 6% | 19% | 8% |
| Pop | 12% | 6% | 0% |
| Rap | 6% | 0% | 6% |
| Electronic | 5% | 0% | 8% |
| Jazz | 4% | 0% | 0% |
| Country | 2% | 6% | 0% |
| Reggae | 0% | 0% | 4% |
| unknown (missing from the MUSDB README) | 0% | 0% | 2% |

- The test set is very unbalanced: 72% Pop/Rock and **no** Rock, Singer/Songwriter, Pop or Jazz. It also has no MedleyDB tracks (only DSD, 2 Native Instruments and 2 from the Heise contest). The MedleyDB tracks (46% of train+val) are only used for training.
- Reggae only appears in test (2 tracks) and Jazz only in train (3). Per-genre test scores rest on 2 to 4 tracks for every genre except Pop/Rock (36), so I only read them as orders of magnitude.
- Artists: 3 artists are in both train and test (*Hollow Ground*, *Skelpolu*, *The Long Wait*), i.e. 3 test tracks out of 50.
- Durations: median around 4 min in all three subsets. Train has a few very short tracks (*Music Delta*, 13 to 26 s); the longest is 10.5 min.

![artists](../figures/dataset_artists.png)
