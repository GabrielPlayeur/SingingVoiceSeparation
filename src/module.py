import torch
import torch.nn as nn
import lightning as L
from .models.unet import UNet
from .utils.utils import si_sdr
from .utils.input_transform import INPUT_TRANSFORMS, compress, transform_amplitude
from .utils.losses import LOSS_TYPES, sisdr_loss, wsdr_loss, multires_stft_loss

SCHEDULERS = ("none", "cosine", "wsd")


class VoiceSeparationModule(L.LightningModule):
    def __init__(self, 
                 base_channels: int, 
                 depth: int, 
                 n_fft: int, 
                 hop_length: int,
                 learning_rate: float,
                 sr: int,
                 in_channels: int = 1,
                 input_transform: str = "none",  # see utils/input_transform.py; absent from old checkpoints = "none"
                 input_power: float = 0.3,       # exponent of input_transform="power"
                 loss_power: float = 1.0,        # compression exponent in the loss; 1.0 = L1 on the linear amplitude (baseline)
                 loss_type: str = "amp_l1",      # see utils/losses.py; absent from old checkpoints = "amp_l1" (baseline)
                 legacy_head: bool = False,      # True: BN + ReLU in the last decoder block (bug fixed in commit 68ba08f), see models/unet.py; absent from old checkpoints = False
                 scheduler: str = "none",        # "none": constant lr (baseline) | "cosine": cosine decay to 0 | "wsd": constant lr then linear decay to 0 at the end; absent from old checkpoints = "none"
                 decay_fraction: float = 0.2,    # scheduler="wsd": final share of the budget (in steps) over which the lr decays linearly to 0
                ):
        super().__init__()
        if input_transform not in INPUT_TRANSFORMS:
            raise ValueError(f"unknown input_transform={input_transform!r}, choose among {INPUT_TRANSFORMS}")
        if scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler={scheduler!r}, choose among {SCHEDULERS}")
        if not 0 < decay_fraction <= 1:
            raise ValueError(f"decay_fraction={decay_fraction} must be in ]0, 1]")
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"unknown loss_type={loss_type!r}, choose among {LOSS_TYPES}")
        if loss_type != "amp_l1" and loss_power != 1.0:
            raise ValueError(f"loss_power={loss_power} only has an effect for loss_type='amp_l1' (loss_type={loss_type!r})")
        self.save_hyperparameters()
        self.model = UNet(base_channels=base_channels, depth=depth, in_channels=in_channels, legacy_head=legacy_head)
        self.register_buffer("window", torch.hann_window(n_fft))

    def _stft(self, waveform):
        # (B, C, T) -> (B, C, F, T'); torch.stft only accepts 1D/2D so the channels are flattened
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
        # estimated amplitude + mixture phase -> waveform (B, C, length)
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
        if "vocals" in batch:  # absent at inference
            batch["voc_amp"] = self._stft(batch["vocals"]).abs()
        return batch

    def forward(self, mix_amp):
        return self.model(transform_amplitude(mix_amp, self.hparams.input_transform, self.hparams.input_power))

    def _loss(self, batch, voc_est_amp, voc_est_wave=None):
        """Training and validation loss. `amp_l1`: L1 between compressed amplitudes (Braun & Tashev 2020, Wilson et al.
        2018; loss_power = 1: the baseline's linear L1). The others are computed on the reconstructed waveform (see
        utils/losses.py). Warning: val/loss is only comparable between models with the same loss_type and loss_power (the
        monitored metric is val/si_sdr_median, common to all)."""
        kind = self.hparams.loss_type
        if kind == "amp_l1":
            c = self.hparams.loss_power
            return nn.functional.l1_loss(compress(voc_est_amp, c), compress(batch["voc_amp"], c))
        if voc_est_wave is None:
            voc_est_wave = self._to_wave(voc_est_amp, batch["mix_phase"], batch["vocals"].shape[-1])
        if kind == "sisdr":
            return sisdr_loss(voc_est_wave, batch["vocals"])
        if kind == "wsdr":
            return wsdr_loss(batch["mixture"], batch["vocals"], voc_est_wave)
        return multires_stft_loss(voc_est_wave, batch["vocals"])

    def training_step(self, batch, batch_idx):
        mask = self.forward(batch["mix_amp"])
        voc_est_amp = mask * batch["mix_amp"]
        loss = self._loss(batch, voc_est_amp)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_sisdr, self._val_sisdr_mix = [], []

    def validation_step(self, batch, batch_idx):
        mask = self.forward(batch["mix_amp"])
        voc_est_amp = mask * batch["mix_amp"]
        voc_est_wave = self._to_wave(voc_est_amp, batch["mix_phase"], batch["vocals"].shape[-1])
        loss = self._loss(batch, voc_est_amp, voc_est_wave)


        # SI-SDR per sample (B,), averaged over the channels: the mean weighted by batch_size gives the exact mean over the crops
        n = batch["vocals"].shape[0]
        sisdr = si_sdr(voc_est_wave, batch["vocals"])
        sisdr_mix = si_sdr(batch["mixture"], batch["vocals"])

        self._val_sisdr.append(sisdr.detach())
        self._val_sisdr_mix.append(sisdr_mix.detach())
        self.log("val/loss", loss, on_epoch=True, batch_size=n)  # the only batched metric

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.scheduler == "none":
            return optimizer
        # Cosine of Loshchilov & Hutter 2017 (SGDR, eq. 5) WITHOUT restart: η_t = η_max/2 · (1 + cos(π t / T)), η_min = 0, updated
        # at every step (as in the paper), T = total number of training steps (max_epochs x steps per epoch).
        total_steps = self.trainer.estimated_stepping_batches
        if not 0 < total_steps < float("inf"):
            raise ValueError(f"scheduler={self.hparams.scheduler!r} needs a finite max_epochs: the schedule spans the whole training")
        total_steps = int(total_steps)
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)
        else:
            # WSD without a warmup phase (constant lr from step 0, like the baseline): constant lr, then linear cooldown to 0
            # over the last `decay_fraction` of the steps (Hägele et al. 2024, arXiv:2405.18392: constant lr + cooldown;
            # the name "WSD": Hu et al. 2024, MiniCPM, arXiv:2404.06395). The factor is 1 until step T(1-f), then (T-t)/(T f).
            # ⚠ the value 0.2 of decay_fraction comes from the re-run plan, not checked in the text of the paper.
            decay_steps = max(1, round(total_steps * self.hparams.decay_fraction))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda t: min(1.0, (total_steps - t) / decay_steps))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
