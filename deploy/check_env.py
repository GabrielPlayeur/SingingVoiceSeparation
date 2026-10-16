#!/usr/bin/env python3
"""Smoke test to run before any run: machine, one dataset track, a forward/backward, throughput and VRAM peak.

    python deploy/check_env.py [--device auto|cpu|cuda] [--steps N] [--batch-size B] [--vram-cap-gb 4]

Works on CPU (the device used is displayed prominently). `--vram-cap-gb 4` caps the process's VRAM to check that the
model fits in the project's 4 GB (an OOM = failure). Exit code 1 if a check fails.
"""
import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import REPO_ROOT, cpu_cores, load_env, mem_total_gb

sys.path.insert(0, str(REPO_ROOT))
import torch  # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  ECHEC : {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=None, help="measured steps (default: 30 on GPU, 3 on CPU)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: the config's on GPU, 2 on CPU")
    ap.add_argument("--vram-cap-gb", type=float, default=None)
    args = ap.parse_args()
    env = load_env()

    print("== Machine ==")
    print(f"Python {platform.python_version()} | torch {torch.__version__} | CUDA (build torch) {torch.version.cuda}")
    print(f"CPU: {cpu_cores(env)} cores (nproc={os.cpu_count()}) | total RAM: {mem_total_gb():.1f} GiB")
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg or 'ABSENT'}")
    if not ffmpeg:
        fail("ffmpeg not in PATH (apt install ffmpeg): preprocessing and museval need it")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        fail("--device cuda requested but torch.cuda.is_available() is False (incompatible driver / CUDA, instance without GPU)")
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\n" + "#" * 60 + f"\n# DEVICE USED: {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else "  (no GPU: functional test only)") + "\n" + "#" * 60)
    if use_cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM {total:.1f} GiB | compute capability {torch.cuda.get_device_capability(0)}")
        if args.vram_cap_gb:
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total))
            print(f"VRAM capped at {args.vram_cap_gb} GiB for this test")

    from hydra import compose, initialize_config_dir
    from src.data.data import list_track_dirs
    from src.data.dataset import MUSDBTrainDataset
    from src.module import VoiceSeparationModule

    processed = env["MUSDB_PROCESSED"]
    if not os.path.isdir(processed):
        processed = str(REPO_ROOT / "musdb18" / "processed")  # local machine: relative path of the config
    with initialize_config_dir(config_dir=str(REPO_ROOT / "src" / "configs"), version_base=None):
        cfg = compose("config", overrides=["data.channels=2", f"data.dataset_path={processed}"])
    batch_size = args.batch_size or (cfg.data.batch_size if use_cuda else 2)
    steps = args.steps or (30 if use_cuda else 3)

    print("\n== Data ==")
    batch = None
    try:
        track_dirs = list_track_dirs(processed, "train")
        ds = MUSDBTrainDataset(track_dirs=track_dirs[:1], crop_length=cfg.data.crop_length, sr=cfg.data.sr, epoch_size=batch_size,
                               silence_threshold=cfg.data.silence_threshold, max_retries=cfg.data.max_retries, channels=2)
        t0 = time.perf_counter()
        items = [ds[i] for i in range(batch_size)]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        print(f"track: {os.path.basename(track_dirs[0])} | {len(track_dirs)} train tracks in {processed} | "
              f"{batch_size} crops {tuple(batch['mixture'].shape[1:])} read in {time.perf_counter() - t0:.2f} s")
    except Exception as e:  # dataset missing or unreadable: continue with a synthetic batch to test the rest
        fail(f"cannot read the dataset ({type(e).__name__}: {e}): run deploy/prepare_data.sh")
        batch = {k: torch.randn(batch_size, 2, cfg.data.crop_length) * 0.1 for k in ("mixture", "vocals")}
        print("  (synthetic batch used for the rest)")

    print("\n== Forward / backward ==")
    module = VoiceSeparationModule(**cfg.model, sr=cfg.data.sr).to(device)
    optimizer = module.configure_optimizers()
    batch = module.on_after_batch_transfer({k: v.to(device) for k, v in batch.items()}, 0)
    n_params = sum(p.numel() for p in module.parameters())
    print(f"U-Net: {n_params / 1e6:.2f} M parameters | batch {batch_size} | input {tuple(batch['mix_amp'].shape)}")

    def step():
        optimizer.zero_grad(set_to_none=True)
        mask = module(batch["mix_amp"])
        loss = module._loss(batch, mask * batch["mix_amp"])  # same computation as training_step, without self.log outside a Trainer
        loss.backward()
        optimizer.step()
        return loss

    try:
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        loss = step()  # warm-up (cuDNN, allocations)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            loss = step()
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if not torch.isfinite(loss):
            fail(f"loss non finie ({loss.item()})")
        print(f"loss {loss.item():.4f} | {steps / dt:.2f} it/s over {steps} steps (local ref. GTX 1650: ~3.5 it/s)")
        if use_cuda:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"peak VRAM (allocated): {peak:.2f} GiB {'< 4 GiB: OK for the \"4 GB\" story' if peak < 4 else '>= 4 GiB: does NOT fit in 4 GB'}")
            if peak >= 4:
                fail(f"VRAM peak {peak:.2f} GiB >= 4 GiB")
    except torch.cuda.OutOfMemoryError:
        fail("GPU out of memory (OOM) during the forward/backward")

    print("\n" + ("ALL OK" if not FAILURES else f"{len(FAILURES)} FAILURE(S):\n- " + "\n- ".join(FAILURES)))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
