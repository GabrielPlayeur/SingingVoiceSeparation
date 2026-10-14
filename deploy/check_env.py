#!/usr/bin/env python3
"""Smoke test à lancer avant tout run : machine, un morceau du dataset, un forward/backward, débit et pic de VRAM.

    python deploy/check_env.py [--device auto|cpu|cuda] [--steps N] [--batch-size B] [--vram-cap-gb 4]

Fonctionne sur CPU (le device utilisé est affiché en gros). `--vram-cap-gb 4` plafonne la VRAM du processus pour vérifier que le
modèle tient dans les 4 Go du projet (une OOM = échec). Code de sortie 1 si une vérification échoue.
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
    ap.add_argument("--steps", type=int, default=None, help="pas mesurés (défaut : 30 sur GPU, 3 sur CPU)")
    ap.add_argument("--batch-size", type=int, default=None, help="défaut : celui de la config sur GPU, 2 sur CPU")
    ap.add_argument("--vram-cap-gb", type=float, default=None)
    args = ap.parse_args()
    env = load_env()

    print("== Machine ==")
    print(f"Python {platform.python_version()} | torch {torch.__version__} | CUDA (build torch) {torch.version.cuda}")
    print(f"CPU : {cpu_cores(env)} cœurs (nproc={os.cpu_count()}) | RAM totale : {mem_total_gb():.1f} Gio")
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg : {ffmpeg or 'ABSENT'}")
    if not ffmpeg:
        fail("ffmpeg absent du PATH (apt install ffmpeg) : le prétraitement et museval en ont besoin")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        fail("--device cuda demandé mais torch.cuda.is_available() est False (pilote / CUDA incompatible, instance sans GPU)")
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\n" + "#" * 60 + f"\n# DEVICE UTILISÉ : {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else "  (aucun GPU : test fonctionnel seulement)") + "\n" + "#" * 60)
    if use_cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU : {torch.cuda.get_device_name(0)} | VRAM {total:.1f} Gio | capacité de calcul {torch.cuda.get_device_capability(0)}")
        if args.vram_cap_gb:
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total))
            print(f"VRAM plafonnée à {args.vram_cap_gb} Gio pour ce test")

    from hydra import compose, initialize_config_dir
    from src.data.data import list_track_dirs
    from src.data.dataset import MUSDBTrainDataset
    from src.module import VoiceSeparationModule

    processed = env["MUSDB_PROCESSED"]
    if not os.path.isdir(processed):
        processed = str(REPO_ROOT / "musdb18" / "processed")  # poste local : chemin relatif de la config
    with initialize_config_dir(config_dir=str(REPO_ROOT / "src" / "configs"), version_base=None):
        cfg = compose("config", overrides=["data.channels=2", f"data.dataset_path={processed}"])
    batch_size = args.batch_size or (cfg.data.batch_size if use_cuda else 2)
    steps = args.steps or (30 if use_cuda else 3)

    print("\n== Données ==")
    batch = None
    try:
        track_dirs = list_track_dirs(processed, "train")
        ds = MUSDBTrainDataset(track_dirs=track_dirs[:1], crop_length=cfg.data.crop_length, sr=cfg.data.sr, epoch_size=batch_size,
                               silence_threshold=cfg.data.silence_threshold, max_retries=cfg.data.max_retries, channels=2)
        t0 = time.perf_counter()
        items = [ds[i] for i in range(batch_size)]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        print(f"morceau : {os.path.basename(track_dirs[0])} | {len(track_dirs)} morceaux train dans {processed} | "
              f"{batch_size} extraits {tuple(batch['mixture'].shape[1:])} lus en {time.perf_counter() - t0:.2f} s")
    except Exception as e:  # dataset absent ou illisible : on continue avec un batch synthétique pour tester le reste
        fail(f"lecture du dataset impossible ({type(e).__name__}: {e}) : exécuter deploy/prepare_data.sh")
        batch = {k: torch.randn(batch_size, 2, cfg.data.crop_length) * 0.1 for k in ("mixture", "vocals")}
        print("  (batch synthétique utilisé pour la suite)")

    print("\n== Forward / backward ==")
    module = VoiceSeparationModule(**cfg.model, sr=cfg.data.sr).to(device)
    optimizer = module.configure_optimizers()
    batch = module.on_after_batch_transfer({k: v.to(device) for k, v in batch.items()}, 0)
    n_params = sum(p.numel() for p in module.parameters())
    print(f"U-Net : {n_params / 1e6:.2f} M paramètres | batch {batch_size} | entrée {tuple(batch['mix_amp'].shape)}")

    def step():
        optimizer.zero_grad(set_to_none=True)
        mask = module(batch["mix_amp"])
        loss = module._loss(batch, mask * batch["mix_amp"])  # même calcul que training_step, sans self.log hors Trainer
        loss.backward()
        optimizer.step()
        return loss

    try:
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        loss = step()  # échauffement (cuDNN, allocations)
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
        print(f"loss {loss.item():.4f} | {steps / dt:.2f} it/s sur {steps} pas (réf. locale GTX 1650 : ~3,5 it/s)")
        if use_cuda:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"pic de VRAM (allouée) : {peak:.2f} Gio {'< 4 Gio : OK pour le récit « 4 Go »' if peak < 4 else '>= 4 Gio : ne tient PAS dans 4 Go'}")
            if peak >= 4:
                fail(f"pic de VRAM {peak:.2f} Gio >= 4 Gio")
    except torch.cuda.OutOfMemoryError:
        fail("mémoire GPU insuffisante (OOM) pendant le forward/backward")

    print("\n" + ("TOUT EST OK" if not FAILURES else f"{len(FAILURES)} ÉCHEC(S) :\n- " + "\n- ".join(FAILURES)))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
