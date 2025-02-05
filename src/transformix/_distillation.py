import os
from typing import Any, Dict, Iterable, Literal, Optional, Union

import pytorch_lightning as pl
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup


class TransformixDistill(pl.LightningModule):
    """
        Pipeline for protein Language Distillation
    """
        
    
    def __init__(
        self, 
        teacher_model: Union[AutoModelForCausalLM, str],
        student_model: Union[AutoModelForCausalLM, str],
        student_config: Union[AutoConfig, Dict[str, Any]] = None,
        lr: float = 1e-3,
        beta1: float = 0.99,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10000,
        num_warmup_steps: int = 1_000,
        freeze: bool = True,
        max_length: int = 512,
    ):
        """Transformix Distillation

        Args:
            teacher_model (Union[AutoModelForCausalLM, str]): Huggingface repository id or AutoModelForCausalLM.
            student_model (Union[AutoModelForCausalLM, str]): Huggingface repository id or AutoModelForCausalLM.
            student_config (Union[AutoConfig, Dict[str, Any]]): student configuration. Default to None.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            beta1 (float, optional): AdamW beta 1. Defaults to 0.99.
            beta2 (float, optional): AdamW beta 2. Defaults to 0.98.
            eps (float, optional): AdamW eps. Defaults to 1e-12.
            num_training_steps (int, optional): maximum training steps. Defaults to 10_000.
            num_warmup_steps (int, optional): number of warmup steps. Defaults to 1_000.
            freeze (bool, optional): freeze teacher or not. Defaults to True.
            max_length (int, optional): max sequence length. Defaults to 512.
        """
        super().__init__()
        
        self._student_config = student_config
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._freeze = freeze
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._max_length = max_length

        if not (isinstance(teacher_model, str) or isinstance(teacher_model, AutoModelForCausalLM)):
            raise ValueError(
                f'teacher_model must be either a string (path to a model) or an instance of AutoModelForCausalLM. '
                f'Got {type(teacher_model)} instead.'
            )
        
        if isinstance(teacher_model, str):
            self._teacher_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(teacher_model, trust_remote_code=True)
        else:
            self._teacher_model = teacher_model
        
        
        if student_model is not None:
            if not (isinstance(student_model, str) or isinstance(student_model, AutoModelForCausalLM)):
                raise ValueError(
                    f'student_model must be either a string (path to a model) or an instance of AutoModelForCausalLM. '
                    f'Got {type(student_model)} instead.'
                )
            elif isinstance(student_model, str):
                self._student_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(student_model, trust_remote_code=True)
            else:
                self._student_model = student_model
        else:
            if student_config is None:
                raise ValueError(
                    f'either `student_model` or `student_config` should be defined. Both are `None`'
                )
            else:
                self._student_model: PreTrainedModel = AutoModelForCausalLM.from_config(student_config)
        
        # Always load tokenizer from teacher
        self._tokenizer = AutoTokenizer.from_pretrained(self._student_model.config._name_or_path, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        if self._freeze:
            self._teacher_model = self._teacher_model.requires_grad_(False)
        
        self.config = self._student_model.config
        
        self.kl_div_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self.ce_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.save_hyperparameters(logger=False)
    
    
    def configure_optimizers(self):
        """Optimizer config is copied from: Cramming Protein Language Model Training in 24 GPU Hours
        
        Link: https://github.com/prescient-design/lobster/blob/main/src/lobster/model/_mlm.py#L218 
        """
        optimizer = torch.optim.AdamW(
            self._student_model.parameters(),
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

    def _compute_loss(self, batch, **kwargs):
        # torch.cuda.empty_cache()
        vocab_size = self._student_model.config.vocab_size
        tokens = batch["input_ids"].squeeze(1)
        attention_mask = batch['attention_mask'].squeeze(1)
        labels = tokens.clone()

        # prepare labels for causal language modeling
        labels[:, :-1] = tokens[:, 1:]
        labels[labels == self._tokenizer.eos_token_id] = -100 # Here eos_token = pad_token for open-ended generation
        
        # teacher forward pass 
        teacher_output = self._teacher_model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # student forward pass
        student_output = self(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        teacher_logits = teacher_output['logits']
        student_logits = student_output['logits']
        
        # distillation loss
        dist_loss = self.kl_div_fn(
            input=torch.log_softmax(student_logits.view(-1, vocab_size), dim=-1),
            target=torch.softmax(teacher_logits.view(-1, vocab_size), dim=-1)
        )
        
        # causal language modeling loss
        clm_loss = self.ce_fn(
            input=student_logits.view(-1, vocab_size),
            target=labels.view(-1) 
        )
    
        loss = dist_loss + clm_loss
        import pdb; pdb.set_trace()
        
        return loss
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):

        return self._student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        train_loss = self._compute_loss(batch=batch)
        return {'loss': train_loss}
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)
    
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        self._student_model.save_pretrained(save_directory, *args, **kwargs)
        self._tokenizer.save_pretrained(save_directory, *args, **kwargs)
