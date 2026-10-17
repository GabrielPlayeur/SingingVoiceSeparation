"""Multichannel Wiener filter as post-processing of a U-Net checkpoint (phase 4 of docs/RERUN_PLAN.md), evaluated with the exact
pipeline of `src.evaluate` (50 test tracks, stereo museval v4, official mixture by default).

Why: Open-Unmix gains 0.16 dB (paired median, 6.06 -> 6.43 dB) from one EM iteration of its Wiener filter (src/evaluate_umx.py). This
measures the same post-processing on our model, without retraining.

How: inside each inference window of `separate_full_track` (5.93 s, 512 frames, 50 % overlap, Hann overlap-add, so memory does not
grow with the track length), the U-Net mask m gives the two source models |V| = m·|X| (vocals) and |A| = (1 − m)·|X| (accompaniment,
the complement, as in the evaluation); `openunmix.filtering.wiener` (one EM iteration, `softmask=False`, Open-Unmix's default) re-estimates
the vocals STFT from the mixture STFT, then inverse STFT. The window plays the role of Open-Unmix's `wiener_win_len` (300 frames there).
Two sources only, where Open-Unmix filters its four targets estimated independently.
A first version filtered the whole-track STFT and was killed by the out-of-memory killer on long tracks (7 GB machine).

Usage (openunmix is not a dependency of the project, as for evaluate_umx.py):
    uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_wiener CKPT -w 0 -o outputs/w32_wiener.json   # -w 0: metrics after each track, a 7 GB machine runs out of memory otherwise
"""
import argparse

import torch

from .evaluate import evaluate
from .inference import separate_full_track
from .module import VoiceSeparationModule


class WienerModule(VoiceSeparationModule):
    """Same network; `predict_step` adds the Wiener filter, so `separate_full_track` (chunks + overlap-add) is reused unchanged."""
    niter = 1

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        from openunmix.filtering import wiener  # imported here: only needed with this script
        X = self._stft(batch["mixture"])                       # (B, C, F, T') complex
        mag = X.abs()
        mask = self.forward(mag)
        voc = []
        for b in range(X.shape[0]):
            spec = torch.stack([mask[b] * mag[b], (1 - mask[b]) * mag[b]], dim=-1).permute(2, 1, 0, 3)  # (T', F, C, 2 sources)
            # floor: bins where both source models are 0 (digital silence) gave a 0/0 and a NaN estimate on one track
            # (Punkdisco - Oral Hygiene) in the first full run; 1e-8 is far below any real magnitude (choice of this script)
            spec = spec.clamp_min(1e-8)
            mix = torch.view_as_real(X[b].permute(2, 1, 0).contiguous())                                   # (T', F, C, 2 real/imag)
            y = wiener(spec, mix, self.niter, softmask=False, residual=False)                             # (T', F, C, 2, 2 sources)
            voc.append(torch.view_as_complex(y[..., 0].contiguous()).permute(2, 1, 0))                    # (C, F, T')
        V = torch.stack(voc)
        B, C = V.shape[:2]
        wave = torch.istft(V.reshape(B * C, *V.shape[-2:]), n_fft=self.hparams.n_fft, hop_length=self.hparams.hop_length,
                           window=self.window, length=batch["mixture"].shape[-1])
        return wave.reshape(B, C, -1)


def make_separate_fn(ckpt, niter=1, crop_length=261632):
    module = WienerModule.load_from_checkpoint(ckpt, map_location="cpu").eval()
    module.niter = niter
    return lambda mixture: separate_full_track(module, mixture, crop_length, batch_size=4)


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
