from typing import List, Optional, Union
from pydantic import BaseModel

from transformix.config._components import (
    DataModuleConfig,
    LightningLoggerConfig,
    LightningCallbackConfig,
    LightningTrainerConfig
)


class MaskedLanguageModelConfig(BaseModel):    
    model_name: str
    lr: float
    beta1: float
    beta2: float
    eps: float
    num_training_steps: int
    num_warmup_steps: int
    freeze: bool
    initial_mask_percentage: float
    mask_percentage: float
    max_length: int
    position_embedding_type: str



class MaskedLanguageModelTrainerConfig(BaseModel):
    model: MaskedLanguageModelConfig
    datamodule: DataModuleConfig
    callbacks: Union[List[LightningCallbackConfig], None] = []
    loggers: Union[List[LightningLoggerConfig], None] = []
    trainer: LightningTrainerConfig
