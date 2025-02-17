import os
from typing import Iterable, Literal, Optional, Union

import pytorch_lightning as pl
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup

from transformix import CodexConfig


class TransformixCodex(pl.LightningModule):
    def __init__(self,
        protein_model: Optional[Union[AutoModelForCausalLM, str]] = None, 
        language_model: Optional[Union[AutoModelForCausalLM, str]] = None,
        lr: float = 1e-3,
        beta1: float = 0.99,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10000,
        num_warmup_steps: int = 1000,
        freeze: bool = False,
        max_length: int = 512,
    ):
        """Transformix Codex Model

        Args:
            protein_model (Optional[Union[AutoModelForCausalLM, str]], optional): protein language model. Defaults to None.
            language_model (Optional[Union[AutoModelForCausalLM, str]], optional): language model. Defaults to None.
            lr (float, optional): AdamW learning rate. Defaults to 1e-3.
            beta1 (float, optional): AdamW beta1. Defaults to 0.99.
            beta2 (float, optional): AdamW beta2. Defaults to 0.98.
            eps (float, optional): AdamW eps. Defaults to 1e-12.
            num_training_steps (int, optional): number of maximum training steps. Defaults to 10000.
            num_warmup_steps (int, optional): number of warmup steps. Defaults to 1000.
            freeze (bool, optional): _description_. Defaults to False.
            max_length (int, optional): max input length. Defaults to 512.
        """
        super().__init__()
        
        if protein_model is None:
            raise ValueError("Protein model cannot be None. Please provide a valid model or model name.")

        if language_model is None:
            raise ValueError("Language model cannot be None. Please provide a valid model or model name.")

        if isinstance(protein_model, str):
            self.protein_model: PreTrainedModel = AutoModel.from_pretrained(protein_model)
        else:
            self.protein_model: PreTrainedModel = protein_model

        if isinstance(language_model, str):
            self.language_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(language_model)
        else:
            self.language_model: PreTrainedModel = language_model
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_length = max_length
        # self.freeze = freeze
        self.config = CodexConfig(
            protein_config=self.protein_model.config,
            text_config=self.language_model.config,
            ignore_index=-100,
            protein_token_id=None,
            projector_hidden_act='gelu',
            protein_seq_length=max_length,
            multimodal_projector_bias=False
        )
        
        self.save_hyperparameters(logger=False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._num_training_steps,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
    def forward(self,):
        pass
    
    
    
