import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .data import MUSDB18DataModule
from .module import VoiceSeparationModule
from .evaluate import evaluate


@hydra.main(version_base=None,
            config_path="configs",
            config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    datamodule = MUSDB18DataModule(**cfg.data)

    module = VoiceSeparationModule(**cfg.model, sr=cfg.data.sr)


    logger = WandbLogger(
        project="VoiceSeparation",
        entity="playeur-audio-team",
        name=cfg.get("wandb", {}).get("run_name", None),
        config=dict(cfg),
        save_dir=cfg.output_dir,
        )

    logger.experiment.define_metric("val/si_sdr_median", summary="max")
    logger.experiment.define_metric("val/si_sdri_median", summary="max")

    callbacks = [
        EarlyStopping(
            monitor="val/si_sdr_median",
            mode="max",
            patience=cfg.training.early_stopping_patience,
            min_delta=cfg.training.early_stopping_min_delta,
        ),
        ModelCheckpoint(monitor="val/si_sdr_median", save_top_k=1, mode="max"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=cfg.output_dir,
    )

    trainer.fit(module, datamodule)

    if cfg.training.evaluate_at_end:
        evaluate(
            trainer.checkpoint_callback.best_model_path,
            dataset_path=cfg.data.dataset_path,
            crop_seconds=cfg.data.crop_seconds,
            logger=logger,
        )


if __name__ == "__main__":
    main()