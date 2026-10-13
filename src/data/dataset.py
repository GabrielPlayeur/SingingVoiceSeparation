import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

def read_crop(path, start, n_frames, sr: int, channels: int=1):
    """Lit un extrait d'un WAV sans charger le fichier entier."""
    x, track_sr = sf.read(path, start=start, frames=n_frames, dtype="float32", always_2d=True)
    if track_sr != sr:
        raise ValueError(f"The wav file {path} is at {track_sr} but the model needs {sr}")
    return to_channels(x, channels)  # (channels, n_frames)

def to_channels(x, channels: int):
    """(n, 2) -> (channels, n) : mono = moyenne des deux canaux, stéréo = tel quel."""
    if channels == 1:
        x = x.mean(axis=1, keepdims=True)
    elif channels != x.shape[1]:
        raise ValueError(f"channels={channels} but the wav file has {x.shape[1]} channels")
    return x.T

class MUSDBTrainDataset(Dataset):
    def __init__(self,
                 track_dirs: str,
                 crop_length: int,
                 sr: int,
                 epoch_size: int,
                 silence_threshold: float,
                 max_retries: int,
                 channels: int = 1,
                ):
        super().__init__()
        self.track_dirs = track_dirs
        self.crop_length = crop_length # en échantillons
        self.sr = sr
        self.epoch_size = epoch_size
        self.silence_threshold = silence_threshold
        self.max_retries = max_retries
        self.channels = channels

        # on lit la durée de chaque morceau une seule fois, au démarrage
        self.track_lengths = []
        for d in track_dirs:
            info = sf.info(os.path.join(d, "mixture.wav"))
            self.track_lengths.append(info.frames)

        n_positions = np.array(self.track_lengths) - self.crop_length
        self.track_weights = n_positions / n_positions.sum()

    def __len__(self):
        return self.epoch_size  # pas de longueur naturelle, on en choisit une

    def _sample_crop(self):
        idx = np.random.choice(len(self.track_dirs),  p=self.track_weights)
        track_dir = self.track_dirs[idx]
        max_start = self.track_lengths[idx] - self.crop_length
        start = np.random.randint(0, max_start)

        vocals = read_crop(os.path.join(track_dir, "vocals.wav"), start, self.crop_length, self.sr, self.channels)
        accompaniment = read_crop(os.path.join(track_dir, "accompaniment.wav"), start, self.crop_length, self.sr, self.channels)
        return vocals, accompaniment

    def __getitem__(self, index):
        vocals, accompaniment = None, None
        # rejette les extraits sans voix (silence), jusqu'à max_retries essais
        for i in range(self.max_retries):
            vocals, accompaniment = self._sample_crop()
            if np.abs(vocals).mean() > self.silence_threshold:
                break
        else:
            print(f"Max retries ({self.max_retries}) reached, keeping a silent crop")

        mixture = vocals + accompaniment
        return {
            "mixture": torch.from_numpy(mixture).float(),
            "vocals": torch.from_numpy(vocals).float(),
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
    """Charge un morceau ENTIER, pour l'évaluation finale."""

    def __init__(self, track_dirs, sr, channels: int = 1):
        self.sr = sr
        self.channels = channels
        self.track_dirs = track_dirs

    def __len__(self):
        return len(self.track_dirs)

    def __getitem__(self, index):
        track_dir = self.track_dirs[index]
        vocals, vocals_sr = sf.read(os.path.join(track_dir, "vocals.wav"), dtype="float32", always_2d=True)
        if vocals_sr != self.sr:
            raise ValueError(f"The wav file {os.path.join(track_dir, 'vocals.wav')} is at {vocals_sr} but the model needs {self.sr}")
        accompaniment, accompaniment_sr  = sf.read(os.path.join(track_dir, "accompaniment.wav"), dtype="float32", always_2d=True)
        if accompaniment_sr != self.sr:
            raise ValueError(f"The wav file {os.path.join(track_dir, 'accompaniment.wav')} is at {accompaniment_sr} but the model needs {self.sr}")
        vocals = to_channels(vocals, self.channels)  # (channels, T)
        accompaniment = to_channels(accompaniment, self.channels)
        return {
            "mixture": torch.from_numpy(vocals + accompaniment).float(),
            "vocals": torch.from_numpy(vocals).float(),
            "track_name": os.path.basename(track_dir),
        }