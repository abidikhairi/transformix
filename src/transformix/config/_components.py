from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field


class GenericObjectConfig(BaseModel):
    name: str = Field(..., alias='name')
    class_path: str = Field(..., alias='classPath')
    args: Dict[str, Any] = Field(..., alias='args')

class LightningCallbackConfig(GenericObjectConfig):
    pass

class LightningLoggerConfig(GenericObjectConfig):
    pass

class DataModuleConfig(GenericObjectConfig):
    pass

class LightningTrainerConfig(BaseModel):
    accelerator: str
    devices: int
    num_nodes: int
    accumulate_grad_batches: int
    gradient_clip_val: float
    gradient_clip_algorithm: str
    precision: str
    max_steps: int
    limit_val_batches: int
    num_sanity_val_steps: int
    max_time: Optional[Union[str, None]]  # Allowing None or string
    val_check_interval: int
    log_every_n_steps: int
    enable_model_summary: Optional[bool] = True
    enable_progress_bar: Optional[bool] = True
    strategy: Optional[str] = 'ddp_find_unused_parameters_true'
