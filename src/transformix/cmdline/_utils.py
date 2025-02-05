import argparse
import importlib
from typing import Any, Dict, List, Type
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from pytorch_lightning import LightningDataModule

from transformix.config._components import (
    GenericObjectConfig,
    LightningCallbackConfig,
    LightningLoggerConfig
)
from transformix.logging_config import logger


def instantiate_object_from_dict(clazz: Type, kwargs: Dict[str, Any]):
    """Instantiates any passed object"""
    
    logger.info(f'Instantiating object of type: {clazz}')
    
    return clazz(**kwargs)

def instantiate_object_from_generic_config(config: GenericObjectConfig):
    """Instantiates any object that inherit `transformix.config._components.GenericObjectConfig`"""
    logger.info(f'Instantiating {config.name} from {config.class_path}')
    
    module_path, class_name = config.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)

    cls = getattr(module, class_name)
    try:
        instance = cls(**config.args)
        logger.info(f'Object {config.name} instantiated successfully')
        
        return instance
    except Exception as e:
        logger.error(f'Could not instantiate {config.name} object from {config.class_path}')
        raise Exception(f'Could not instantiate {config.name} object from {config.class_path}, exception: {e}')

def instantiate_callbacks(callbacks_cfg: List[LightningCallbackConfig]) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg or len(callbacks_cfg) == 0:
        logger.info("[instantiate_callbacks] No callback configs found! Skipping..")
        return callbacks
    
    for cb_cfg in callbacks_cfg:
        logger.info(f"[instantiate_callbacks] Instantiating callback <{cb_cfg.name}>")
        
        class_path = cb_cfg.class_path
        module_path, class_name = class_path.rsplit(".", 1)

        module = importlib.import_module(module_path)

        cls = getattr(module, class_name)

        instance: Callback = cls(**cb_cfg.args)
        logger.info(f"[instantiate_callbacks] callback <{cb_cfg.name}> instantiated successfully")

        callbacks.append(instance)
        
    return callbacks

def instantiate_loggers(loggers_cfg: List[LightningLoggerConfig]) -> list[Callback]:
    """Instantiates loggers from config."""
    loggers: list[Callback] = []

    if not loggers_cfg or len(loggers_cfg) == 0:
        logger.info("[instantiate_loggers] No logger configs found! Skipping..")
        return loggers
    
    for logger_cfg in loggers_cfg:
        logger.info(f"[instantiate_loggers] Instantiating logger <{logger_cfg.name}>")
        
        class_path = logger_cfg.class_path
        module_path, class_name = class_path.rsplit(".", 1)
        
        module = importlib.import_module(module_path)

        cls = getattr(module, class_name)

        instance: Logger = cls(**logger_cfg.args)

        logger.info(f"[instantiate_loggers] Logger <{logger_cfg.name}> instantiated successfully")
        
        loggers.append(instance)
        
    return loggers


def instantiate_datamodule(datamodule_cfg: Dict[str, Any]) -> LightningDataModule:
    if not datamodule_cfg or datamodule_cfg == {}:
        print('[instantiate_datamodule] No datamodule configs found!')
        raise ValueError('datamodule config is not defined, check you config.json file')
    class_path = datamodule_cfg['classPath']
    module_path, class_name = class_path.rsplit(".", 1)
        
    module = importlib.import_module(module_path)

    cls = getattr(module, class_name)
    
    datamodule: LightningDataModule = cls(**datamodule_cfg['args'])

    return datamodule

def get_cli_args(description: str, args=None):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--experiment-config-path', type=str, required=True, help='Path to the experiment config file (mlm.json, distillation.json, clm.json, etc)')

    return parser.parse_args() if args is None else parser.parse_args(args)
