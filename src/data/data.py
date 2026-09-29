import json
import os
from typing import Any

import numpy as np
import lightning as L
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
import soundfile as sf

from .dataset import MUSDBTrainDataset, MUSDBFixedDataset, MUSDBFullTrackDataset, read_crop


def list_track_dirs(dataset_path, split="train"):
    base = os.path.join(dataset_path, split)
    return sorted(
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )


class MUSDB18DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        sr: int,
        crop_seconds: float,
        batch_size: int,
        epoch_size: int,
        val_crops_per_track: int,
        val_size: float,
        silence_threshold: float,
        max_retries: int,
        num_workers: int,
        random_seed: int,
        split_path: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.crop_length = int(sr * crop_seconds)

    def _get_track_dirs(self, split="train"):
        return list_track_dirs(self.hparams.dataset_path, split)

    def _load_split(self, train_dirs):
        # split par artiste, sauvegardé pour être reproductible
        if os.path.exists(self.hparams.split_path):
            with open(self.hparams.split_path) as f:
                saved = json.load(f)
            # if the params have changed
            if (saved["params"]["val_size"] != self.hparams.val_size or 
                saved["params"]["random_seed"] != self.hparams.random_seed or 
                saved["params"]["dataset_path "] != self.hparams.dataset_path 
                ):
                return self._build_split(train_dirs)
            return saved["train"], saved["val"]
        return self._build_split(train_dirs)

    def _build_split(self, train_dirs):
        # le nom d'artiste = tout avant " - " dans le nom du dossier
        names = [os.path.basename(d) for d in train_dirs]
        groups = [n.split(" - ")[0] for n in names]

        splitter = GroupShuffleSplit(n_splits=1, test_size=self.hparams.val_size, random_state=self.hparams.random_seed)
        train_idx, val_idx = next(splitter.split(train_dirs, groups=groups))

        params = {"val_size": self.hparams.val_size, "random_seed": self.hparams.random_seed, "dataset_path": self.hparams.dataset_path}
        train_split = [train_dirs[i] for i in train_idx]
        val_split = [train_dirs[i] for i in val_idx]

        os.makedirs(os.path.dirname(self.hparams.split_path), exist_ok=True)
        with open(self.hparams.split_path, "w") as f:
            json.dump({"params": params, "train": train_split, "val": val_split}, f, indent=2)

        return train_split, val_split

    def _build_val_crops(self, val_dirs):
        rng = np.random.default_rng(self.hparams.random_seed)
        crops = []
        for track_dir in val_dirs:
            n_frames = sf.info(os.path.join(track_dir, "mixture.wav")).frames
            kept, tries = 0, 0
            while kept < self.hparams.val_crops_per_track and tries < 50:
                tries += 1
                start = int(rng.integers(0, n_frames - self.crop_length))
                vocals = read_crop(os.path.join(track_dir, "vocals.wav"), start, self.crop_length, sr=self.hparams.sr)
                if np.abs(vocals).mean() < 1e-4:  # skip silence, redo
                    continue
                accompaniment = read_crop(os.path.join(track_dir, "accompaniment.wav"), start, self.crop_length, sr=self.hparams.sr)
                crops.append({"mixture": vocals + accompaniment, "vocals": vocals})
                kept += 1
        return crops

    def setup(self, stage: str) -> None:
        all_train_dirs = self._get_track_dirs("train")
        train_dirs, val_dirs = self._load_split(all_train_dirs)

        if stage == "fit" or stage is None:
            self.train_dataset = MUSDBTrainDataset(
                track_dirs=train_dirs,
                crop_length=self.crop_length,
                sr=self.hparams.sr,
                epoch_size=self.hparams.epoch_size,
                silence_threshold=self.hparams.silence_threshold,
                max_retries=self.hparams.max_retries,
            )
            val_crops = self._build_val_crops(val_dirs)
            self.val_dataset = MUSDBFixedDataset(val_crops)

        if stage == "test" or stage is None:
            self.test_dataset = MUSDBFullTrackDataset(
                track_dirs=self._get_track_dirs("test"),
                sr=self.hparams.sr
                )

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> Any:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )