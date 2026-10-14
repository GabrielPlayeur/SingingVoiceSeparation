import torch
import torch.nn as nn
import lightning as L
from .models.unet import UNet
from .utils.utils import si_sdr
from .utils.input_transform import INPUT_TRANSFORMS, transform_amplitude

class VoiceSeparationModule(L.LightningModule):
    def __init__(self, 
                 base_channels: int, 
                 depth: int, 
                 n_fft: int, 
                 hop_length: int,
                 learning_rate: float,
                 sr: int,
                 in_channels: int = 1,
                 input_transform: str = "none",
                ):
        super().__init__()
        if input_transform not in INPUT_TRANSFORMS:
            raise ValueError(f"input_transform={input_transform!r} inconnu, choisir parmi {INPUT_TRANSFORMS}")
        self.save_hyperparameters()
        self.model = UNet(base_channels=base_channels, depth=depth, in_channels=in_channels)
        self.register_buffer("window", torch.hann_window(n_fft))

    def _stft(self, waveform):
        # (B, C, T) -> (B, C, F, T') ; torch.stft n'accepte que 1D/2D donc on aplatit les canaux
        B, C, T = waveform.shape
        spec = torch.stft(
            waveform.reshape(B * C, T),
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        return spec.reshape(B, C, *spec.shape[-2:])

    def _to_wave(self, amp, phase, length):
        # amplitude est + phase mix -> forme d'onde (B, C, length)
        spec = amp * torch.exp(1j * phase)
        B, C = spec.shape[:2]
        wave = torch.istft(
            spec.reshape(B * C, *spec.shape[-2:]),
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            window=self.window,
            length=length,
        )
        return wave.reshape(B, C, length)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        mix_stft = self._stft(batch["mixture"])
        batch["mix_amp"] = mix_stft.abs()
        batch["mix_phase"] = mix_stft.angle()
        if "vocals" in batch:  # absent en inference
            batch["voc_amp"] = self._stft(batch["vocals"]).abs()
        return batch

    def forward(self, mix_amp):
        return self.model(transform_amplitude(mix_amp, self.hparams.input_transform))

    def training_step(self, batch, batch_idx):
        mask = self.forward(batch["mix_amp"])
        voc_est_amp = mask * batch["mix_amp"]
        loss = nn.functional.l1_loss(voc_est_amp, batch["voc_amp"])
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_sisdr, self._val_sisdr_mix = [], []

    def validation_step(self, batch, batch_idx):
        mask = self.forward(batch["mix_amp"])
        voc_est_amp = mask * batch["mix_amp"]
        loss = nn.functional.l1_loss(voc_est_amp, batch["voc_amp"])

        voc_est_wave = self._to_wave(voc_est_amp, batch["mix_phase"], batch["vocals"].shape[-1])


        # SI-SDR par échantillon (B,), moyenné sur les canaux : la moyenne pondérée par batch_size donne la moyenne exacte sur les crops
        n = batch["vocals"].shape[0]
        sisdr = si_sdr(voc_est_wave, batch["vocals"])
        sisdr_mix = si_sdr(batch["mixture"], batch["vocals"])

        self._val_sisdr.append(sisdr.detach())
        self._val_sisdr_mix.append(sisdr_mix.detach())
        self.log("val/loss", loss, on_epoch=True, batch_size=n)  # seule métrique batchée

    def on_validation_epoch_end(self):
        s = torch.cat(self._val_sisdr)
        m = torch.cat(self._val_sisdr_mix)
        self.log("val/si_sdr_median", torch.quantile(s, 0.5))
        self.log("val/si_sdr_mean", s.mean())
        self.log("val/si_sdr_mix_median", torch.quantile(m, 0.5))
        self.log("val/si_sdri_median", torch.quantile(s - m, 0.5))
        self.log("val/si_sdri_mean", (s - m).mean())


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        voc_amp = self.forward(batch["mix_amp"]) * batch["mix_amp"]
        return self._to_wave(voc_amp, batch["mix_phase"], batch["mixture"].shape[-1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)