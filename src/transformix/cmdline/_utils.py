import importlib
from typing import Any, Dict, List
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

from pytorch_lightning import LightningDataModule


def instantiate_callbacks(callbacks_cfg: List[Dict[str, Dict[str, Any]]]) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg or len(callbacks_cfg) == 0:
        print("[instantiate_callbacks] No callback configs found! Skipping..")
        return callbacks
    
    for cb_cfg in callbacks_cfg:
        print(f"[instantiate_callbacks] Instantiating callback <{cb_cfg['name']}>")
        
        class_path = cb_cfg['classPath']
        module_path, class_name = class_path.rsplit(".", 1)

        module = importlib.import_module(module_path)

        cls = getattr(module, class_name)
        
        instance: Callback = cls(**cb_cfg['args'])
        callbacks.append(instance)
        
    return callbacks

def instantiate_loggers(loggers_cfg: List[Dict[str, Dict[str, Any]]]) -> list[Callback]:
    """Instantiates loggers from config."""
    loggers: list[Callback] = []

    if not loggers_cfg or len(loggers_cfg) == 0:
        print("[instantiate_loggers] No logger configs found! Skipping..")
        return loggers
    
    for logger_cfg in loggers_cfg:
        print(f"[instantiate_loggers] Instantiating logger <{logger_cfg['name']}>")
        
        class_path = logger_cfg['classPath']
        module_path, class_name = class_path.rsplit(".", 1)
        
        module = importlib.import_module(module_path)

        cls = getattr(module, class_name)
        
        instance: Logger = cls(**logger_cfg['args'])
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
