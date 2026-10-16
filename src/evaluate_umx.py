"""
Calibration point: evaluates the pretrained Open-Unmix `umx` (Stöter et al. 2019, JOSS) with the exact pipeline of `src.evaluate`
(same 50 test tracks, same stereo museval v4 protocol, same bootstrap CI), to check that the pipeline reproduces a published number
before trusting the comparisons with our U-Net.

Usage (openunmix is not a dependency of the project: it is pulled in an ephemeral environment, torch stays at 2.10.0+cu128):
    uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_umx --niter 0 -o outputs/umx_niter0.json
    uv run --with openunmix --with torchaudio==2.10.0 python -m src.evaluate_umx --niter 1 -o outputs/umx_niter1.json

`--niter 0`: |V| estimated by the vocals network times the phase of the mixture (the same reconstruction as our U-Net, but the
             magnitude is an unbounded ReLU output, not a sigmoid mask).
`--niter 1`: Open-Unmix's default; one EM iteration of the multichannel Wiener filter (Nugraha et al. 2016; norbert) over the
             FOUR targets (bass, drums, other, vocals) as in the published setup. The gain is the Wiener filter's own contribution.
The accompaniment estimate is mixture - vocals estimate, as for the U-Net: only the vocals SDR is compared.
"""
import argparse

import torch

from .evaluate import evaluate


def separate_vocals(separator, mixture, niter, device):
    """Vocals estimate (2, T) of an `openunmix.model.Separator`, with the memory of ONE target instead of four.

    Same computation as `Separator.forward` (STFT, one network per target, `wiener` over windows of `wiener_win_len` frames, ISTFT),
    but only the vocals channel of the filtered output is kept: `Separator.forward` allocates the STFT of the 4 targets over the
    whole track, ~5 GB of RAM for a 6 min track, which got the first attempt of this evaluation killed by the OOM killer (7.8 GB
    of RAM shared with the metric workers). With `niter=0` the other three networks are skipped: the masked-phase estimate of the
    vocals does not depend on them (`softmask=False`)."""
    from openunmix.filtering import wiener
    names = list(separator.target_models)
    vi = names.index("vocals")
    needed = [vi] if niter == 0 else range(len(names))
    sep = separator.to(device)
    with torch.no_grad():
        mix_stft = sep.stft(mixture[None].to(device))  # (1, 2, bins, frames, 2)
        X = sep.complexnorm(mix_stft)  # (1, 2, bins, frames)
        mix_stft = mix_stft.cpu()
        specs = {j: sep.target_models[names[j]](X.detach().clone()).cpu() for j in needed}  # (1, 2, bins, frames) each
        del X
        n_frames = mix_stft.shape[3]
        out = mix_stft  # the vocals estimate overwrites the mixture STFT window by window (each window is read before it is written)
        win = sep.wiener_win_len or n_frames
        for p in range(0, n_frames, win):
            q = min(n_frames, p + win)
            spec = torch.stack([specs[j][0, :, :, p:q] for j in needed], dim=-1).permute(2, 1, 0, 3)  # (w, bins, 2, S)
            mix = mix_stft[0, :, :, p:q].permute(2, 1, 0, 3)  # (w, bins, 2, 2)
            y = wiener(spec.to(device), mix.to(device), niter, softmask=sep.softmask, residual=sep.residual)  # (w, bins, 2, 2, S)
            out[0, :, :, p:q] = y[..., list(needed).index(vi)].permute(2, 1, 0, 3).cpu()  # (w, bins, 2, 2) -> (2, bins, w, 2)
        est = sep.istft(out[:, None].to(device), length=mixture.shape[-1])  # (1, 1, 2, T)
    return est[0, 0].cpu()


def make_separate_fn(niter, device="cuda", model="umx"):
    from openunmix.utils import load_separator  # imported here: only needed with this script
    separator = load_separator(model, niter=niter, device="cpu")
    separator.freeze()
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    def separate(mixture):  # (2, T) at 44.1 kHz -> (2, T)
        try:
            return separate_vocals(separator, mixture, niter, device)
        except torch.cuda.OutOfMemoryError:  # 4 GB of VRAM, possibly shared with another job: this track goes through the CPU
            torch.cuda.empty_cache()
            return separate_vocals(separator, mixture, niter, "cpu")
    return separate


def check_against_separator(niter, model="umx", seconds=40, track="musdb18/processed/test/AM Contra - Heart Peripheral"):
    """Compares `separate_vocals` with the reference `Separator.forward` on an excerpt (must be identical up to float rounding)."""
    import soundfile as sf
    from openunmix.utils import load_separator
    x, _ = sf.read(f"{track}/mixture.wav", dtype="float32", always_2d=True, start=44100 * 60, frames=44100 * seconds)
    mixture = torch.from_numpy(x.T.copy())
    separator = load_separator(model, niter=niter, device="cpu")
    separator.freeze()
    with torch.no_grad():
        ref = separator(mixture[None])[0, list(separator.target_models).index("vocals")]
    mine = separate_vocals(separator, mixture, niter, "cpu")
    print(f"niter={niter}: max |ref - mine| = {(ref - mine).abs().max():.2e}  (max |ref| = {ref.abs().max():.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, choices=(0, 1), required=True)
    parser.add_argument("--model", default="umx", help="umx (MUSDB18), umxhq (MUSDB18-HQ), umxl")
    parser.add_argument("--dataset-path", default="musdb18/processed")
    parser.add_argument("--mixture", choices=("sum", "official"), default="official",
                        help="'official' (default) = mixture.wav, the protocol of the literature; 'sum' = vocals + accompaniment")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("-o", "--output-json", default=None)
    parser.add_argument("-w", "--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--check", action="store_true", help="only compares separate_vocals with Separator.forward on 40 s")
    args = parser.parse_args()
    if args.check:
        check_against_separator(args.niter, args.model)
        raise SystemExit
    evaluate(f"{args.model}_niter{args.niter}", args.dataset_path, max_tracks=args.max_tracks, output_json=args.output_json,
             workers=args.workers, mixture=args.mixture, separate_fn=make_separate_fn(args.niter, args.device, args.model),
             partial_path=f"outputs/{args.model}_niter{args.niter}_{args.mixture}.partial.jsonl")  # rerun the same command to resume
