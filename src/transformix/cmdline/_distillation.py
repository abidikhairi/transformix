# This file is part of the transformix project:
# It run an MLM model on a dataset
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from pydantic import ValidationError
import torch
import pytorch_lightning as pl

from transformix.logging_config import logger
from transformix import TransformixDistill
from transformix.config import DistillationTrainerConfig
from transformix.cmdline._utils import (
    instantiate_callbacks,
    instantiate_loggers,
    get_cli_args,
    instantiate_object_from_dict,
    instantiate_object_from_generic_config
)


# Set float32 matmul precision to medium: Laptop GPUs (RTX family) do not support high precision
torch.set_float32_matmul_precision('medium')


def distillation(args=None):
    if args is None:
        args = get_cli_args(description="Runs knowledge distillation pipeline")
    
    experiment_config_path = args.experiment_config_path
    
    logger.info(f'Running Distillation with experiment config file: {experiment_config_path}')
    logger.info(f'Parsing configuration file')
    
    try:
        trainer_config = DistillationTrainerConfig.model_validate_json(Path(experiment_config_path).read_text())
        model: TransformixDistill = instantiate_object_from_dict(TransformixDistill, trainer_config.model.model_dump())
        datamodule = instantiate_object_from_generic_config(trainer_config.datamodule)
        callbacks = instantiate_callbacks(trainer_config.callbacks)
        loggers = instantiate_loggers(trainer_config.loggers)
        
        trainer_args = trainer_config.trainer.model_dump()
        
        trainer = pl.Trainer(
            **trainer_args,
            callbacks=callbacks,
            logger=loggers,
        )
        
        trainer.fit(model, datamodule)
        
    except ValidationError as e:
        logger.error(f'Cannot parse file {experiment_config_path}, due to {e.errors()}')


if __name__ == '__main__':    
    args = get_cli_args(description="Runs knowledge distillation pipeline")

    distillation(args)
