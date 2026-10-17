"""Listening demo: separated excerpts of a test track, with spectrograms, and the static page that serves them.

Two steps:
  extract CKPT TRACK_DIR   separates the WHOLE track (same inference as src.evaluate: 5.93 s crops, 50 % overlap, official mixture),
                           scores every 1 s window (BSSEval v4, the statistic of the evaluation) and writes two 10 s excerpts
                           chosen by RULE, not by ear: `best` and `worst`.
  page                     rebuilds demo/index.html from the demo/*/*/info.json files.

Excerpt rule (so that a good passage does not hide a bad one): candidate excerpts are the 10 s windows starting every 5 s with at
least 8 finite 1 s SDRs AND at least 8 one-second windows with vocals (RMS within 40 dB of the track's 90th percentile; the
-40 dB is my choice, not sourced). The vocal filter was added after a first run whose `worst` excerpt was a near-silent
vocal passage (SDR -15 dB on a reference at the noise floor: a metric artefact, not an audible failure); the score of a candidate is the median of its 1 s SDRs (the
evaluation's statistic restricted to the excerpt). `best` = highest score, `worst` = lowest. A
`typical` excerpt (closest to the median) was dropped from the page: three excerpts per track made it too long to listen to. The choice looks at the ground truth: it is a description of the model, not a blind test.

Only tracks whose licence allows sharing may be published (MUSDB18 README: CC BY-NC-SA 3.0 for the two Easton Ellises tracks of
the test set, the others are "Restricted"). Files of an excerpt (demo/<track>/<kind>/): mix, voice_est, voice_true, accomp_est
(.mp3 128 kbps by default, or .wav with --format wav; 44.1 kHz stereo, one common gain so that nothing clips) and spectrogram.jpg.

Usage:
    uv run python -m analysis.make_demo extract outputs/VoiceSeparation/fgvzjxua/checkpoints/baseline-stereo.ckpt \
        "musdb18/processed/test/The Easton Ellises - Falcon 69"
    uv run python -m analysis.make_demo page
"""
import argparse
import html
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

from ._common import *

DEMO_DIR = os.path.join(ROOT, "demo")
SECONDS = 10         # excerpt length
HOP_S = 5            # candidate excerpts start every 5 s
MIN_FINITE = 8       # of the 10 one-second SDRs, and of the 10 one-second windows with vocals
ACTIVE_DB = -40      # a 1 s window has vocals if its RMS is within 40 dB of the track's 90th percentile (see window_scores)
KINDS = ("best", "worst")
FILES = ("mix", "voice_est", "voice_true", "accomp_est")
LICENCE = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
TRACKS = {  # tracks allowed for publication: name -> (slug, attribution)
    "The Easton Ellises - Falcon 69": ("falcon-69", "The Easton Ellises, “Falcon 69”"),
    "The Easton Ellises (Baumi) - SDRNR": ("sdrnr", "The Easton Ellises (Baumi), “SDRNR”"),
}


def active_windows(vocals, sr, floor_db=ACTIVE_DB):
    """(T, C) true vocals -> bool per 1 s window: RMS within `floor_db` of the 90th percentile of the 1 s RMS of the track."""
    n = len(vocals) // sr
    rms = np.sqrt((vocals[:n * sr].reshape(n, sr, -1) ** 2).mean(axis=(1, 2)))
    return rms > np.percentile(rms, 90) * 10 ** (floor_db / 20)


def window_scores(sdr, active, seconds=SECONDS, hop=HOP_S, min_finite=MIN_FINITE):
    """sdr: 1 s SDRs of the track; active: 1 s windows with vocals. Returns [(start_s, median SDR)] for the candidate excerpts."""
    out = []
    for k in range(0, len(sdr) - seconds + 1, hop):
        w = sdr[k:k + seconds]
        if np.isfinite(w).sum() >= min_finite and active[k:k + seconds].sum() >= min_finite:
            out.append((k, float(np.nanmedian(w))))
    return out


def pick(cands):
    """{'best', 'worst'} -> (start_s, score), by the rule of the docstring."""
    scores = np.array([s for _, s in cands])
    return {"best": cands[int(scores.argmax())], "worst": cands[int(scores.argmin())]}


def encode(path_wo_ext, x, sr, fmt):
    """x: (T, C) float. Writes .wav, or .mp3 (ffmpeg, libmp3lame 128 kbps)."""
    wav = path_wo_ext + ".wav"
    sf.write(wav, x, sr, subtype="PCM_16")
    if fmt == "mp3":
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "128k", path_wo_ext + ".mp3"],
                       check=True)
        os.remove(wav)


def spectrogram_figure(signals, sr, path):
    """Log-magnitude spectrograms (STFT 2048 / hop 512 like the model), mono mix of the channels, common scale."""
    import scipy.signal
    setup_style()
    fig, axes = plt.subplots(len(signals), 1, figsize=(9, 2.4 * len(signals)), sharex=True, constrained_layout=True)
    ref = None
    for ax, (title, x) in zip(np.atleast_1d(axes), signals.items()):
        f, t, z = scipy.signal.stft(x.mean(axis=1), fs=sr, nperseg=2048, noverlap=2048 - 512)
        db = 20 * np.log10(np.abs(z) + 1e-8)
        ref = db.max() if ref is None else ref   # first panel (the mixture) sets the scale
        m = f <= 16000                           # AAC stems are low-passed near 16 kHz
        im = ax.pcolormesh(t, f[m] / 1000, db[m], vmin=ref - 80, vmax=ref, cmap="magma", shading="auto", rasterized=True)
        ax.set_ylabel("kHz"); ax.set_title(title, loc="left", fontsize=10); ax.grid(False)
    axes[-1].set_xlabel("time in the excerpt (s)")
    fig.colorbar(im, ax=axes, label="dB (relative to the mixture's maximum)", shrink=0.8)
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)


def extract(args):
    import torch
    import lightning as L
    from src.data.dataset import MUSDBFullTrackDataset
    from src.inference import separate_full_track
    from src.module import VoiceSeparationModule
    from src.utils.fast_museval import evaluate as bsseval
    from src.utils.utils import assert_no_temporal_padding

    name = os.path.basename(os.path.normpath(args.track))
    if name not in TRACKS and not args.unlisted:
        sys.exit(f"{name!r} is not in the publishable list {sorted(TRACKS)} (MUSDB18 licences). --unlisted for a private test.")
    slug = TRACKS.get(name, (name.lower().replace(" ", "-"), name))[0]

    module = VoiceSeparationModule.load_from_checkpoint(args.ckpt, map_location="cpu").eval()
    sr, crop = module.hparams.sr, 261632
    assert_no_temporal_padding(crop, module.hparams.hop_length, module.hparams.depth)
    s = MUSDBFullTrackDataset(track_dirs=[args.track], sr=sr, channels=2, mixture="official")[0]
    mixture, vocals, accomp = s["mixture"], s["vocals"], s["accompaniment"]
    trainer = L.Trainer(accelerator="cpu" if args.cpu else "auto", devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False)
    with torch.no_grad():
        v_est = separate_full_track(module, mixture, crop, 16, trainer)
    a_est = mixture - v_est
    mix, v_est, v_true, a_est = (t.numpy().T for t in (mixture, v_est, vocals, a_est))   # (T, C)

    def stack(a, b):
        return np.stack([a, b])                                                           # (2, T, C)
    sdr, _, sir, sar = bsseval(stack(v_true, accomp.numpy().T), stack(v_est, a_est), win=sr, hop=sr)
    cands = window_scores(sdr[0], active_windows(v_true, sr))
    track_sdr = float(np.nanmedian(sdr[0]))
    print(f"{name}: track median SDR {track_sdr:.2f} dB over {np.isfinite(sdr[0]).sum()} windows of 1 s, {len(cands)} candidate excerpts")

    for kind, (k0, score) in pick(cands).items():
        a, b = k0 * sr, (k0 + SECONDS) * sr
        parts = dict(mix=mix[a:b], voice_est=v_est[a:b], voice_true=v_true[a:b], accomp_est=a_est[a:b])
        gain = min(1.0, 0.98 / max(np.abs(p).max() for p in parts.values()))               # common gain: no clipping, same levels
        out = os.path.join(DEMO_DIR, slug, kind)
        os.makedirs(out, exist_ok=True)
        for f, x in parts.items():
            encode(os.path.join(out, f), x * gain, sr, args.format)
        spectrogram_figure({"Mixture": parts["mix"], "Estimated vocals": parts["voice_est"], "True vocals": parts["voice_true"]},
                           sr, os.path.join(out, "spectrogram.jpg"))
        info = dict(track=name, slug=slug, attribution=TRACKS.get(name, (None, name))[1], kind=kind, start_s=k0, seconds=SECONDS,
                    sdr=score, sir=float(np.nanmedian(sir[0][k0:k0 + SECONDS])), sar=float(np.nanmedian(sar[0][k0:k0 + SECONDS])),
                    track_sdr=track_sdr, ckpt=os.path.relpath(args.ckpt, ROOT), format=args.format, gain=float(gain))
        with open(os.path.join(out, "info.json"), "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"  {kind:8s} from {k0:3d} s: SDR {info['sdr']:+.2f}  SIR {info['sir']:+.2f}  SAR {info['sar']:+.2f} dB -> {os.path.relpath(out, ROOT)}")


def page(args):
    infos = []
    for slug, _ in TRACKS.values():
        for kind in KINDS:
            p = os.path.join(DEMO_DIR, slug, kind, "info.json")
            if os.path.exists(p):
                infos.append(json.load(open(p)))
    if not infos:
        sys.exit("no demo/<track>/<kind>/info.json: run `extract` first")
    e = html.escape
    body = []
    for slug in dict.fromkeys(i["slug"] for i in infos):
        group = [i for i in infos if i["slug"] == slug]
        body.append(f"<h2>{e(group[0]['attribution'])}</h2><p class=meta>Median SDR of the whole track: {group[0]['track_sdr']:.2f} dB.</p>")
        for i in sorted(group, key=lambda i: KINDS.index(i["kind"])):
            ext = i["format"]
            audio = "".join(f"<figure><figcaption>{label}</figcaption><audio controls preload=none src=\"{slug}/{i['kind']}/{f}.{ext}\"></audio></figure>"
                            for f, label in zip(FILES, ("Mixture", "Estimated vocals", "True vocals", "Estimated accompaniment")))
            body.append(f"<section><h3>{i['kind'].capitalize()} excerpt <span class=meta>({i['start_s']}–{i['start_s'] + i['seconds']} s; "
                        f"SDR {i['sdr']:+.1f} dB)</span></h3>"
                        f"<div class=players>{audio}</div><img src=\"{slug}/{i['kind']}/spectrogram.jpg\" alt=\"spectrograms of the {i['kind']} excerpt\"></section>")
    model = e(args.model) if args.model else f"checkpoint <code>{e(infos[0]['ckpt'])}</code>"
    note = f"<p>{e(args.note)}</p>" if args.note else ""
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice separation demo</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;color:#0b0b0b;background:#fff}}
@media (prefers-color-scheme:dark){{body{{background:#161615;color:#ecebe8}}a{{color:#7db2f2}}}}
.meta{{font-size:.85em;opacity:.7;font-weight:400}} .players{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px 16px}}
figure{{margin:0}} figcaption{{font-size:.9em}} audio{{width:100%}} img{{max-width:100%;margin-top:8px}} section{{margin:24px 0}}
</style></head><body>
<h1>Voice separation demo</h1>
<p>Excerpts of two test tracks separated by the small stereo U-Net of this project ({model}). For each track:
the best and the worst 10 s excerpt, chosen by a fixed rule on the 1 s SDR scores (median over the excerpt), so a good
passage does not hide a bad one. Passages without vocals are left out (their SDR only measures the noise floor). SDR is the BSSEval v4
median over the 1 s windows of the excerpt, as in the evaluation.
The selection uses the ground truth: this describes the model, it is not a blind test.</p>
{note}
{''.join(body)}
<h2>Licence and changes</h2>
<p>Audio from the MUSDB18 dataset (Rafii et al., 2017), tracks by The Easton Ellises from the heise stems remix competition,
licensed under <a href="{LICENCE}">Creative Commons BY-NC-SA 3.0</a>. <b>The audio has been modified</b>: cut to 10 s excerpts,
re-encoded to MP3 128 kbps, and the vocals/accompaniment estimates are outputs of a neural network. The excerpts on this page are
shared under the same licence (attribution, non-commercial, share-alike).</p>
</body></html>
"""
    with open(os.path.join(DEMO_DIR, "index.html"), "w") as fh:
        fh.write(doc)
    print(f"-> demo/index.html ({len(infos)} excerpts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("ckpt"); p.add_argument("track", help="musdb18/processed/test/<track>")
    p.add_argument("--format", choices=("mp3", "wav"), default="mp3")
    p.add_argument("--cpu", action="store_true", help="inference on CPU (free VRAM for other jobs)")
    p.add_argument("--unlisted", action="store_true", help="allow a track outside the publishable list (private test, do not commit)")
    p = sub.add_parser("page")
    p.add_argument("--model", default=None, help="description of the model shown on the page (default: the checkpoint path)")
    p.add_argument("--note", default=None, help="paragraph shown before the excerpts (context, caveats)")
    args = ap.parse_args()
    {"extract": extract, "page": page}[args.cmd](args)


if __name__ == "__main__":
    main()
