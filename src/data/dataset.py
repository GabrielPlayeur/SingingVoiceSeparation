import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

def read_crop(path, start, n_frames, sr: int, channels: int=1):
    """Reads an excerpt of a WAV without loading the whole file."""
    x, track_sr = sf.read(path, start=start, frames=n_frames, dtype="float32", always_2d=True)
    if track_sr != sr:
        raise ValueError(f"The wav file {path} is at {track_sr} but the model needs {sr}")
    return to_channels(x, channels)  # (channels, n_frames)

def to_channels(x, channels: int):
    """(n, 2) -> (channels, n): mono = mean of the two channels, stereo = as is."""
    if channels == 1:
        x = x.mean(axis=1, keepdims=True)
    elif channels != x.shape[1]:
        raise ValueError(f"channels={channels} but the wav file has {x.shape[1]} channels")
    return x.T

class MUSDBTrainDataset(Dataset):
    """Random training crops. Each item is `vocals + accompaniment` (the sum, not mixture.wav) and the vocals.

    Three independent augmentations, all off by default (then the sampling is the one of the original baseline):
      - `remix`: vocals from track A, accompaniment from another track B drawn independently (same weights as A, B may equal A)
        at an independent offset. Same idea as the random track mixing of Open-Unmix (`random_track_mix` in
        openunmix/data.py), but on 2 sources: musdb18/processed only holds `vocals` and `accompaniment`, not the 4 stems.
      - `gain`: each source is multiplied by its own gain, uniform in [`gain_low`, `gain_high`]. The defaults [0.25, 1.25]
        are the ones of `_augment_gain` in openunmix/data.py (checked in the source, branch master).
      - `channel_swap`: each source has its L/R channels swapped with probability 0.5 (`_augment_channelswap`, same file).
        Stereo only.
    Augmentations are applied per source, before the sum, so the mixture is always the exact sum of the (augmented) sources.
    The silence rejection (`silence_threshold`) only looks at the raw vocals crop, before any augmentation: the accompaniment
    may be silent (an intro with only a voice is a legitimate sample), and the gain must not change what "silent" means.
    """
    def __init__(self,
                 track_dirs: str,
                 crop_length: int,
                 sr: int,
                 epoch_size: int,
                 silence_threshold: float,
                 max_retries: int,
                 channels: int = 1,
                 remix: bool = False,
                 gain: bool = False,
                 gain_low: float = 0.25,
                 gain_high: float = 1.25,
                 channel_swap: bool = False,
                ):
        super().__init__()
        if channel_swap and channels != 2:
            raise ValueError("channel_swap needs channels=2 (swapping L/R of a mono signal does nothing)")
        if not 0 < gain_low <= gain_high:
            raise ValueError(f"gain range must satisfy 0 < gain_low <= gain_high, got [{gain_low}, {gain_high}]")
        self.track_dirs = track_dirs
        self.crop_length = crop_length # in samples
        self.sr = sr
        self.epoch_size = epoch_size
        self.silence_threshold = silence_threshold
        self.max_retries = max_retries
        self.channels = channels
        self.remix = remix
        self.gain = gain
        self.gain_low = gain_low
        self.gain_high = gain_high
        self.channel_swap = channel_swap

        # the duration of each track is read once, at start-up
        self.track_lengths = []
        for d in track_dirs:
            info = sf.info(os.path.join(d, "mixture.wav"))
            self.track_lengths.append(info.frames)

        n_positions = np.array(self.track_lengths) - self.crop_length
        self.track_weights = n_positions / n_positions.sum()

    def __len__(self):
        return self.epoch_size  # no natural length, we choose one

    def _draw_position(self):
        """(track index, start sample); tracks are drawn proportionally to their number of possible crops."""
        idx = np.random.choice(len(self.track_dirs),  p=self.track_weights)
        max_start = self.track_lengths[idx] - self.crop_length
        return idx, np.random.randint(0, max_start)

    def _read_source(self, idx, start, name):
        return read_crop(os.path.join(self.track_dirs[idx], f"{name}.wav"), start, self.crop_length, self.sr, self.channels)

    def _augment_source(self, x):
        """Per-source gain and L/R swap, x: (channels, crop_length)."""
        if self.gain:
            x = x * np.random.uniform(self.gain_low, self.gain_high)
        if self.channel_swap and np.random.rand() < 0.5:
            x = x[::-1]
        return x

    def __getitem__(self, index):
        # rejects crops without voice (silence), at most max_retries tries
        for i in range(self.max_retries):
            idx, start = self._draw_position()
            vocals = self._read_source(idx, start, "vocals")
            if np.abs(vocals).mean() > self.silence_threshold:
                break
        else:
            print(f"Max retries ({self.max_retries}) reached, keeping a silent crop")

        if self.remix:
            idx, start = self._draw_position()  # track B and offset, independent of the vocals' ones
        accompaniment = self._read_source(idx, start, "accompaniment")

        vocals, accompaniment = self._augment_source(vocals), self._augment_source(accompaniment)
        mixture = vocals + accompaniment
        return {
            "mixture": torch.from_numpy(np.ascontiguousarray(mixture)).float(),
            "vocals": torch.from_numpy(np.ascontiguousarray(vocals)).float(),
        }

class MUSDBFixedDataset(Dataset):
    def __init__(self, crops):
        super().__init__()
        self.crops = crops # {"mixture": array, "vocals": array}

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        c = self.crops[index]
        return {
            "mixture": torch.from_numpy(c["mixture"]).float(),
            "vocals": torch.from_numpy(c["vocals"]).float(),
        }

class MUSDBFullTrackDataset(Dataset):
    """Loads a WHOLE track, for the final evaluation.

    `mixture`: "sum" (default, the training/validation convention) returns `vocals + accompaniment` as the mixture;
    "official" returns mixture.wav, the mixture shipped with MUSDB18 (encoded separately from the stems, so it differs from the
    sum by a ~4 % residual, see analysis/mixture_gap.py): the protocol of the literature.
    `accompaniment` is always the accompaniment.wav reference."""

    def __init__(self, track_dirs, sr, channels: int = 1, mixture: str = "sum"):
        if mixture not in ("sum", "official"):
            raise ValueError(f"mixture={mixture!r}: expected 'sum' or 'official'")
        self.sr = sr
        self.channels = channels
        self.track_dirs = track_dirs
        self.mixture = mixture

    def __len__(self):
        return len(self.track_dirs)

    def _read(self, track_dir, name):
        path = os.path.join(track_dir, f"{name}.wav")
        x, file_sr = sf.read(path, dtype="float32", always_2d=True)
        if file_sr != self.sr:
            raise ValueError(f"The wav file {path} is at {file_sr} but the model needs {self.sr}")
        return to_channels(x, self.channels)  # (channels, T)

    def __getitem__(self, index):
        track_dir = self.track_dirs[index]
        vocals = self._read(track_dir, "vocals")
        accompaniment = self._read(track_dir, "accompaniment")
        mixture = vocals + accompaniment if self.mixture == "sum" else self._read(track_dir, "mixture")
        return {
            "mixture": torch.from_numpy(mixture).float(),
            "vocals": torch.from_numpy(vocals).float(),
            "accompaniment": torch.from_numpy(accompaniment).float(),
            "track_name": os.path.basename(track_dir),
        }
