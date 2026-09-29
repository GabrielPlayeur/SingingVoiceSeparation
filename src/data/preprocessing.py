"""Convertit MUSDB18 (.stem.mp4) en fichiers wav : mixture, vocals, accompaniment.

Structure des stems (cf. musdb18/README.md) :
    0 mixture | 1 drums | 2 bass | 3 other | 4 vocals
accompaniment = drums + bass + other

Usage :
    uv run python src/data/preprocessing.py
    uv run python src/data/preprocessing.py --input musdb18 --output data/processed --workers 4
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

SAMPLE_RATE = 44100
N_CHANNELS = 2

MIXTURE, DRUMS, BASS, OTHER, VOCALS = range(5)


def get_ffmpeg() -> str:
    """ffmpeg du système si présent, sinon le binaire fourni par imageio-ffmpeg."""
    if path := shutil.which("ffmpeg"):
        return path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def read_stem(mp4_path: Path, stream: int, ffmpeg: str) -> np.ndarray:
    """Décode un flux audio du mp4 en float32 de forme (n_samples, n_channels)."""
    cmd = [
        ffmpeg, "-v", "error", "-i", str(mp4_path),
        "-map", f"0:a:{stream}",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(N_CHANNELS), "-ar", str(SAMPLE_RATE),
        "-",
    ]  # fmt: skip
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, N_CHANNELS)


def process_track(args: tuple[Path, Path]) -> str:
    mp4_path, out_dir = args
    if all((out_dir / f"{n}.wav").exists() for n in ("mixture", "vocals", "accompaniment")):
        return f"skip {mp4_path.name}"

    ffmpeg = get_ffmpeg()
    stems = {i: read_stem(mp4_path, i, ffmpeg) for i in (MIXTURE, DRUMS, BASS, OTHER, VOCALS)}

    # Les flux peuvent différer de quelques échantillons après décodage AAC.
    n = min(len(s) for s in stems.values())
    stems = {i: s[:n] for i, s in stems.items()}

    accompaniment = stems[DRUMS] + stems[BASS] + stems[OTHER]

    out_dir.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_dir / "mixture.wav", SAMPLE_RATE, stems[MIXTURE])
    wavfile.write(out_dir / "vocals.wav", SAMPLE_RATE, stems[VOCALS])
    wavfile.write(out_dir / "accompaniment.wav", SAMPLE_RATE, accompaniment)
    return f"ok {mp4_path.name}"


def preprocess(input_dir: Path, output_dir: Path, workers: int = 1) -> None:
    jobs = []
    for subset in ("train", "test"):
        for mp4_path in sorted((input_dir / subset).glob("*.stem.mp4")):
            track_name = mp4_path.name.removesuffix(".stem.mp4")
            jobs.append((mp4_path, output_dir / subset / track_name))

    if not jobs:
        raise FileNotFoundError(f"Aucun .stem.mp4 trouvé dans {input_dir}/{{train,test}}")

    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            list(tqdm(pool.map(process_track, jobs), total=len(jobs), desc="MUSDB18"))
    else:
        for job in tqdm(jobs, desc="MUSDB18"):
            process_track(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("musdb18"))
    parser.add_argument("--output", type=Path, default=Path("musdb18/processed"))
    parser.add_argument("--workers", type=int, default=1)
    cli = parser.parse_args()
    preprocess(cli.input, cli.output, cli.workers)
