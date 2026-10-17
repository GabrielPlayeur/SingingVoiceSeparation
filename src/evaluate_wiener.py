"""Multichannel Wiener filter as post-processing of a U-Net checkpoint (phase 4 of docs/RERUN_PLAN.md), evaluated with the exact
pipeline of `src.evaluate` (50 test tracks, stereo museval v4, official mixture by default).

Why: Open-Unmix gains 0.16 dB (paired median, 6.06 -> 6.43 dB) from one EM iteration of its Wiener filter (src/evaluate_umx.py). This
measures the same post-processing on our model, without retraining.

How: the U-Net separates the track as in the evaluation (`separate_full_track`); the two estimates (vocals, and mixture - vocals as
the accompaniment) are taken back to the STFT of the model (n_fft 2048, hop 512), their magnitudes are the source models of
`openunmix.filtering.wiener` (one EM iteration, `softmask=False`, as Open-Unmix's default), applied to the mixture STFT by windows
of 300 frames (Open-Unmix's `wiener_win_len`, bounds the memory), then inverse STFT. Two sources only (vocals, accompaniment),
where Open-Unmix filters its four targets.
⚠ The magnitudes come from the STFT of the overlap-added waveform, not from the raw mask x |X| of each window: close but not identical.

Usage (openunmix is not a dependency of the project, as for evaluate_umx.py):
    uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_wiener CKPT -o outputs/w32_wiener.json
"""
import argparse

import torch

from .evaluate import evaluate
from .inference import separate_full_track
from .module import VoiceSeparationModule

WIN = 300  # frames per Wiener window (Open-Unmix default wiener_win_len)


def make_separate_fn(ckpt, niter=1, crop_length=261632, device="cuda"):
    from openunmix.filtering import wiener  # imported here: only needed with this script
    module = VoiceSeparationModule.load_from_checkpoint(ckpt, map_location="cpu").eval()
    n_fft, hop = module.hparams.n_fft, module.hparams.hop_length
    window = torch.hann_window(n_fft)
    device = device if torch.cuda.is_available() else "cpu"

    def stft(x):  # (2, T) -> (frames, bins, 2) complex
        return torch.stft(x, n_fft, hop, window=window, return_complex=True, center=True).permute(2, 1, 0)

    def separate(mixture):  # (2, T) -> (2, T)
        voc = separate_full_track(module, mixture, crop_length, batch_size=8)
        X, V, A = stft(mixture), stft(voc), stft(mixture - voc)
        out = torch.empty_like(X)
        for p in range(0, X.shape[0], WIN):
            q = min(X.shape[0], p + WIN)
            spec = torch.stack([V[p:q].abs(), A[p:q].abs()], dim=-1).to(device)        # (w, bins, 2, 2 sources)
            mix = torch.view_as_real(X[p:q]).to(device)                                  # (w, bins, 2, 2 real/imag)
            y = wiener(spec, mix, niter, softmask=False, residual=False)                 # (w, bins, 2, 2 real/imag, 2 sources)
            out[p:q] = torch.view_as_complex(y[..., 0].contiguous()).cpu()
        return torch.istft(out.permute(2, 1, 0), n_fft, hop, window=window, center=True, length=mixture.shape[-1])
    return separate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--niter", type=int, default=1)
    ap.add_argument("--dataset-path", default="musdb18/processed")
    ap.add_argument("--mixture", choices=("sum", "official"), default="official")
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("-o", "--output-json", default=None)
    ap.add_argument("-w", "--workers", type=int, default=None)
    args = ap.parse_args()
    out = args.output_json or "outputs/wiener.json"
    evaluate(f"{args.ckpt} + wiener niter={args.niter}", args.dataset_path, max_tracks=args.max_tracks, output_json=args.output_json,
             workers=args.workers, mixture=args.mixture, separate_fn=make_separate_fn(args.ckpt, args.niter),
             partial_path=out.replace(".json", f"_{args.mixture}.partial.jsonl"))  # rerun the same command to resume
