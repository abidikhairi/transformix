from typing import List, Optional, Union
from pydantic import BaseModel

from transformix.config._components import (
    DataModuleConfig,
    LightningLoggerConfig,
    LightningCallbackConfig,
    LightningTrainerConfig
)


class CausalLanguageModelConfig(BaseModel):
    model: str
    config: Optional[dict]
    lr: float
    beta1: float
    beta2: float
    eps: float
    num_training_steps: int
    num_warmup_steps: int
    freeze: bool
    max_length: int

    
class CausalLanguageModelTrainerConfig(BaseModel):
    model: CausalLanguageModelConfig
    datamodule: DataModuleConfig
    callbacks: Union[List[LightningCallbackConfig], None] = []
    loggers: Union[List[LightningLoggerConfig], None] = []
    trainer: LightningTrainerConfig
