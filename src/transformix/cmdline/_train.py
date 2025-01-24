import hydra
import torch
import lightning.pytorch as pl
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from transformix.cmdline._utils import instantiate_callbacks

# Set float32 matmul precision to medium: Laptop GPUs (RTX family) do not support high precision
torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def train(cfg: DictConfig) -> tuple[pl.LightningModule, pl.LightningDataModule, list[pl.Callback]]:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # wandb.init(
    #     config=log_cfg,  # type: ignore[arg-type]
    #     project=cfg.logger.project,
    #     entity=cfg.logger.entity,
    #     group=cfg.logger.group,
    #     notes=cfg.logger.notes,
    #     tags=cfg.logger.tags,
    #     name=cfg.logger.get("name"),
    # )

    logger = hydra.utils.instantiate(cfg.logger)
    callbacks = instantiate_callbacks(cfg.get("callbacks", []))
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, pl.loggers.WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg})
    
    # if not cfg.dryrun:
    #     trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)
    # import pdb; pdb.set_trace()
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(), 
        val_dataloaders=datamodule.val_dataloader()
    )

    #     if cfg.run_test:
    #         trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # wandb.finish()

    return model, datamodule, callbacks
