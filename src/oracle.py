import soundfile as sf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import museval
import torch
from lightning.pytorch.callbacks import Callback
from .utils import si_sdr

class Oracle:
    def __init__(self, n_fft, hop_length):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def evaluate(self, mixture, vocals):
        mix_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, pad_mode="reflect")
        voc_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length, pad_mode="reflect")
        mix_amp = np.abs(mix_stft)
        mix_phase = np.angle(mix_stft)

        voc_amp = np.abs(voc_stft)
        mask = voc_amp / (mix_amp + 1e-8)
        mask = np.clip(mask, 0, 1) # 0 (no voice) ; 1 (only voice)

        voc_est_amp = mix_amp * mask
        voc_est_stft = voc_est_amp * np.exp(1j * mix_phase)
        voc_est = librosa.istft(voc_est_stft, hop_length=self.hop_length, length=mixture.shape[-1])  # mixture : (T,) ou (C, T)

        return voc_est

class OracleValReference(Callback):
    def __init__(self, n_fft, hop_length):
        self.oracle = Oracle(n_fft, hop_length)
        self.value = None

    def on_fit_start(self, trainer, pl_module):
        val_crops = trainer.datamodule.val_dataset
        scores = []
        for i in range(len(val_crops)):
            crop = val_crops[i]
            voc_est = self.oracle.evaluate(crop["mixture"].numpy(), crop["vocals"].numpy())
            scores.append(si_sdr(torch.from_numpy(voc_est), crop["vocals"]))
        self.value = torch.quantile(torch.stack(scores), 0.5).item()  # même médiane que le module

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            pl_module.log("val/si_sdr_oracle_median", self.value)

def main():
    SOUND_PATH = "musdb18/processed/train/Actions - One Minute Smile/"

    mix, sr = sf.read(os.path.join(SOUND_PATH, "mixture.wav"))
    vocals, _ = sf.read(os.path.join(SOUND_PATH, "vocals.wav"))
    print(mix.shape, sr) # (n_sample, 2) Hz

    mix_mono = mix.mean(axis=1)
    voc_mono = vocals.mean(axis=1)

    start = sr * 30 # skip intro
    duration = sr * 10 # 10 sec
    mix_crop = mix_mono[start:start+duration]
    voc_crop = voc_mono[start:start+duration]
    print(mix_crop.shape) # (Hz*10, )

    sf.write("test_mix.wav", mix_crop, sr)
    sf.write("test_voc.wav", voc_crop, sr)


    n_fft = 2048
    hop_length = 512

    mix_stft = librosa.stft(mix_crop, n_fft=n_fft, hop_length=hop_length)
    voc_stft = librosa.stft(voc_crop, n_fft=n_fft, hop_length=hop_length)

    mix_amp = np.abs(mix_stft)
    mix_phase = np.angle(mix_stft)
    voc_amp = np.abs(voc_stft)

    # plt.imshow(librosa.amplitude_to_db(mix_amp), origin="lower", aspect="auto")
    # plt.show()

    mask = voc_amp / (mix_amp + 1e-8)
    mask = np.clip(mask, 0, 1) # 0 (no voice) ; 1 (only voice)

    print(mask.shape) # (1025, 431)
    print(mask.min(), mask.max())   # between 0 and 1

    # plt.imshow(mask, origin="lower", aspect="auto")
    # plt.show()

    voc_est_amp = mix_amp * mask
    voc_est_stft = voc_est_amp * np.exp(1j * mix_phase)
    voc_est = librosa.istft(voc_est_stft, hop_length=hop_length)
    sf.write("test_voc_est.wav", voc_est, sr)


    # plt.subplot(2,1,1)
    # plt.imshow(voc_est_amp, origin="lower", aspect="auto")
    # plt.title("EST")
    # plt.subplot(2,1,2)
    # plt.imshow(voc_amp, origin="lower", aspect="auto")
    # plt.title("TRUE")
    # plt.show()


    sdr = museval.evaluate(
        references=voc_crop[np.newaxis, :, np.newaxis],   # (sources, samples, canaux)
        estimates=voc_est[np.newaxis, :len(voc_crop), np.newaxis],
    )
    print(sdr)