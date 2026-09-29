"""
Usage :
    python -m src.inference CKPT INPUT.wav OUTPUT.wav [--crop-seconds 6.0]
"""
import math
import argparse

import soundfile as sf
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader

from .module import VoiceSeparationModule


def separate_full_track(
    module: VoiceSeparationModule,
    mixture: torch.Tensor,
    crop_length: int,
    batch_size: int,
    trainer: L.Trainer | None = None,
) -> torch.Tensor:
    if trainer is None:
        trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

    T = mixture.shape[-1]
    hop = crop_length // 2  # 50% recouvrement

    # on pad la fin pour que les chunks couvrent toute la piste
    n_chunks = max(math.ceil((T - crop_length) / hop), 0) + 1
    padded_len = (n_chunks - 1) * hop + crop_length
    padded = F.pad(mixture, (0, padded_len - T))

    chunks = padded.unfold(0, crop_length, hop)  # (n_chunks, crop_length)
    loader = DataLoader([{"mixture": c} for c in chunks], batch_size=batch_size)
    voc_chunks = torch.cat(trainer.predict(module, loader))

    window = torch.hann_window(crop_length + 2, dtype=torch.float64)[1:-1].float()

    out = torch.zeros(padded_len)
    norm = torch.zeros(padded_len)
    for i, chunk in enumerate(voc_chunks):
        start = i * hop
        out[start:start + crop_length] += chunk * window
        norm[start:start + crop_length] += window

    return (out / norm)[:T]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    module = VoiceSeparationModule.load_from_checkpoint(args.ckpt, map_location="cpu")

    x, sr = sf.read(args.input, dtype="float32", always_2d=True)  # (T, C)
    if sr != module.hparams.sr:
        raise ValueError(f"The wav file is at {sr} but the model needs {module.hparams.sr}")
    mixture = torch.from_numpy(x.mean(axis=1))  # stereo -> mono

    voice = separate_full_track(module, mixture, int(sr * args.crop_seconds), args.batch_size)
    sf.write(args.output, voice.numpy(), sr)


