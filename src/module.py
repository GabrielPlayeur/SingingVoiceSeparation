import torch
import torch.nn as nn
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
import lightning as L
from .models.unet import UNet

class VoiceSeparationModule(L.LightningModule):
    def __init__(self, 
                 base_channels: int, 
                 depth: int, 
                 n_fft: int, 
                 hop_length: int,
                 learning_rate: float,
                 sr: int,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(base_channels=base_channels, depth=depth)
        self.register_buffer("window", torch.hann_window(n_fft))

    def _stft(self, waveform):
        return torch.stft(
            waveform,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            window=self.window,
            return_complex=True,
        )

    def _to_wave(self, amp, phase, length):
        # amplitude est + phase mix -> forme d'onde (B, length)
        return torch.istft(
            amp * torch.exp(1j * phase),
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            window=self.window,
            length=length,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        mix_stft = self._stft(batch["mixture"])
        batch["mix_amp"] = mix_stft.abs()
        batch["mix_phase"] = mix_stft.angle()
        if "vocals" in batch:  # absent en inference
            batch["voc_amp"] = self._stft(batch["vocals"]).abs()
        return batch

    def forward(self, mix_amp):
        # mix_amp : (B, F, T) ->  (B, 1, F, T)
        x = mix_amp.unsqueeze(1)
        mask = self.model(x)
        return mask.squeeze(1)

    def training_step(self, batch, batch_idx):
        mask = self.forward(batch["mix_amp"])
        voc_est_amp = mask * batch["mix_amp"]
        loss = nn.functional.l1_loss(voc_est_amp, batch["voc_amp"])
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mask = self.forward(batch["mix_amp"])
        voc_est_amp = mask * batch["mix_amp"]
        loss = nn.functional.l1_loss(voc_est_amp, batch["voc_amp"])

        voc_est_wave = self._to_wave(voc_est_amp, batch["mix_phase"], batch["vocals"].shape[-1])

        # SI-SDR par échantillon (B,) : la moyenne pondérée par batch_size donne la moyenne exacte sur les crops
        n = batch["vocals"].shape[0]
        sisdr = scale_invariant_signal_distortion_ratio(voc_est_wave, batch["vocals"]).mean()
        sisdr_mix = scale_invariant_signal_distortion_ratio(batch["mixture"], batch["vocals"]).mean()
        self.log("val/loss", loss, on_epoch=True, batch_size=n)
        self.log("val/si_sdr", sisdr, on_epoch=True, batch_size=n)
        self.log("val/si_sdr_mix", sisdr_mix, on_epoch=True, batch_size=n)
        self.log("val/si_sdri", sisdr - sisdr_mix, on_epoch=True, batch_size=n)  # gain

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        voc_amp = self.forward(batch["mix_amp"]) * batch["mix_amp"]
        return self._to_wave(voc_amp, batch["mix_phase"], batch["mixture"].shape[-1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)