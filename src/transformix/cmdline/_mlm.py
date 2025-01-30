# This file is part of the transformix project:
# It run an MLM model on a dataset
import argparse
import json
from typing import Any, Dict
import torch
import pytorch_lightning as pl

from transformix import TransformixPMLM
from transformix.cmdline._utils import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_datamodule
)


# Set float32 matmul precision to medium: Laptop GPUs (RTX family) do not support high precision
torch.set_float32_matmul_precision('medium')


def mlm(args=None):
    if args is None:
        args = get_cli_args()
    
    experiment_config_path = args.experiment_config_path
    
    print(f'Running MLM with experiment config: {experiment_config_path}')
    experiment_config = parse_experiment_config(experiment_config_path)
    model_id = experiment_config['model']['model_name']
    
    data_module = instantiate_datamodule(experiment_config['datamodule'])
    model = TransformixPMLM(**experiment_config['model'])
    
    callbacks = instantiate_callbacks(experiment_config['callbacks'])
    loggers = instantiate_loggers(experiment_config['loggers'])
    
    trainer = pl.Trainer(
        **experiment_config['trainer'],
        callbacks=callbacks,
        logger=loggers
    )

    trainer.fit(model, data_module)
    
    trainer.model.model.push_to_hub(model_id)


def parse_experiment_config(experiment_config_path: str) -> Dict[str, Any]:
    with open(experiment_config_path, 'r') as f:
        experiment_config = json.load(f)
    return experiment_config


def get_cli_args(args=None):
    parser = argparse.ArgumentParser(description='Run an MLM model on a dataset')

    parser.add_argument('--experiment-config-path', type=str, required=True, help='Path to the experiment config file (mlm.json)')

    return parser.parse_args() if args is None else parser.parse_args(args)


if __name__ == '__main__':
    
    args = get_cli_args()

    mlm(args)
