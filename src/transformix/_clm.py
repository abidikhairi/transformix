import os
from typing import Optional, Union

import pytorch_lightning as pl
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup


class TransformixCLM(pl.LightningModule):
    """
        Pipeline for Protein Causal Language Modeling.
    """

    def __init__(
        self, 
        model: Optional[Union[AutoModelForCausalLM, str]] = None,
        config: Optional[AutoConfig] = None,
        lr: float = 1e-3,
        beta1: float = 0.99,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10000,
        num_warmup_steps: int = 1_000,
        freeze: bool = False,
        max_length: int = 512,
    ):
        """Transformix Distillation

        Args:
            model (Optional[Union[AutoModelForCausalLM, str]]): huggingface repository id or AutoModelForCausalLM.
            config (Optional[AutoConfig]): model configuration. Default to None.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            beta1 (float, optional): AdamW beta 1. Defaults to 0.99.
            beta2 (float, optional): AdamW beta 2. Defaults to 0.98.
            eps (float, optional): AdamW eps. Defaults to 1e-12.
            num_training_steps (int, optional): maximum training steps. Defaults to 10_000.
            num_warmup_steps (int, optional): number of warmup steps. Defaults to 1_000.
            freeze (bool, optional): freeze model or not. Defaults to False.
            max_length (int, optional): max sequence length. Defaults to 512.
        """
        super().__init__()
        
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._freeze = freeze
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._max_length = max_length

        if model is None and config is None:
            raise ValueError('Both model and config cannot be None. Please provide at least one.')

        if model is not None:
            if not (isinstance(model, str) or isinstance(model, AutoModelForCausalLM)):
                raise ValueError(
                    f'model must be either a string (path to a model or huggingface repository id) or an instance of `AutoModelForCausalLM`. '
                    f'Got {type(model)} instead.'
                )
        
            if isinstance(model, str):
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
            else:
                self.model = model
        else:
            self.model = AutoModelForCausalLM.from_config(config)
        
        # set model to training mode
        self.model = self.model.train()        
            
        # Always load tokenizer from teacher
        # padding_side is set to right in training/validation
        self._tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True, padding_side='right')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        if self._freeze:
            self._freeze_all_but_lm_head()
        
        self.config = self.model.config
        
        self.ce_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.save_hyperparameters(logger=False)
    
    def _freeze_all_but_lm_head(self):
        for name, param in self.model.named_parameters():
            if "lm_head" not in name:  # Skip the lm head
                param.requires_grad = False
    
    def configure_optimizers(self):
        """Optimizer config is copied from: Cramming Protein Language Model Training in 24 GPU Hours
        
        Link: https://github.com/prescient-design/lobster/blob/main/src/lobster/model/_mlm.py#L218 
        """
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

    def _compute_loss(self, batch, **kwargs):
        # torch.cuda.empty_cache()
        vocab_size = self.config.vocab_size
        
        tokens = batch["input_ids"].squeeze(1)
        attention_mask = batch['attention_mask'].squeeze(1)
        labels = tokens.clone()

        # prepare labels for causal language modeling
        labels[:, :-1] = tokens[:, 1:]
        labels[labels == self._tokenizer.eos_token_id] = -100 # Here eos_token = pad_token for open-ended generation
        
        # student forward pass
        output = self(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        logits = output['logits']
        
        # causal language modeling loss
        loss = self.ce_fn(
            input=logits.view(-1, vocab_size),
            target=labels.view(-1)
        )

        return loss
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch=batch)
        ppl = torch.exp(loss)
        
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/perplexity", ppl, sync_dist=True)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch=batch)
        ppl = torch.exp(loss)
        
        self.log("valid/loss", loss, sync_dist=True)
        self.log("valid/perplexity", ppl, sync_dist=True)

        return {'valid/loss': loss}
    
    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        self.model.save_pretrained(save_directory, *args, **kwargs)
        self._tokenizer.save_pretrained(save_directory, *args, **kwargs)
