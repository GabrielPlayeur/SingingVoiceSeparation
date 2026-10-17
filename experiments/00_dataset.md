# 00 · The dataset: MUSDB18

[← back to README](../README.md)

I use **compressed MUSDB18** (AAC stems), not MUSDB18-HQ: 100 tracks for train/val, 50 for test. Validation is 16 of the 100, split by artist (`GroupShuffleSplit`, `val_size=0.14`, seed 42, saved in `musdb18/split.json`), so no artist is in both train and val.

## Official mixture vs vocals + accompaniment

The stems are compressed separately, so `mixture.wav` isn't exactly `vocals + accompaniment`. I train on the sum, so the input is exactly target + rest.

![mixture gap](../figures/mixture_gap.png)

- The residual is **−30.7 dB** below the mixture (median over 150 tracks), concentrated **above 17 kHz**. My guess is the AAC low-pass; I haven't checked.
- The oracle mask computed on the sum and applied to the real mixture loses **0.05 dB** SI-SDR (10.21 vs 10.17 dB, 48 tracks).
- Evaluating the baseline on the official mixture instead of the sum changes its test SDR by 0.03 dB (4.50 → 4.47, [01](01_baseline_vs_oracle.md)). The README uses the official mixture, like the literature.
- Two tracks are odd: *PR - Oh No* and *PR - Happy Daze*, where the voice is almost inaudible (*Happy Daze*: 67 % of 1 s windows below −60 dB). They come back as outliers on every page.

## Genres and artists

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
| unknown | 0% | 0% | 2% |

- The test set is 72 % Pop/Rock. Outside Pop/Rock, each genre has 2 to 4 test tracks, so per-genre scores are orders of magnitude only.
- 3 artists are in both train and test (3 test tracks out of 50).
- Median duration about 4 min. Train has a few very short tracks (*Music Delta*, 13 to 26 s).

![artists](../figures/dataset_artists.png)

```bash
uv run python -m analysis.mixture_gap
uv run python -m analysis.dataset_distribution
```
