from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from transformix.config._components import (
    DataModuleConfig,
    LightningLoggerConfig,
    LightningCallbackConfig,
    LightningTrainerConfig
)


class ProteinTextLanguageModelConfig(BaseModel):
    protein_encoder: Optional[str] = None
    text_encoder: Optional[str] = None
    adapter: Optional[Dict[str, Any]] = None
    learning_rate: float = 1e-5
    beta1: float = 0.99
    beta2: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 10000
    freeze_protein_encoder: Optional[bool] = True
    freeze_text_encoder: Optional[bool] = False
    freeze_adapter: Optional[bool] = False

class ProteinTextLanguageModelTrainerConfig(BaseModel):
    model: ProteinTextLanguageModelConfig
    datamodule: DataModuleConfig
    callbacks: Union[List[LightningCallbackConfig], None] = []
    loggers: Union[List[LightningLoggerConfig], None] = []
    trainer: LightningTrainerConfig
