import os
from typing import Iterable, Literal, Optional, Union

import pytorch_lightning as pl
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup

from transformix._mlm_configuration import PMLM_CONFIG_ARGS, PMLMConfig
from transformix.lm_base import LMBaseForMaskedLM


class TransformixMLM(pl.LightningModule):
    def __init__(
            self,
            model_name: str = None,
            lr: float = 1e-3,
            beta1: float = 0.9,
            beta2: float = 0.98,
            eps: float = 1e-12,
            num_training_steps: int = 10_000,
            num_warmup_steps: int = 1_000,
            freeze: bool = False,
            mask_percentage: float = 0.15,
            initial_mask_percentage: Optional[float] = None,
            max_length: int = 512,
            position_embedding_type: Literal["rotary", "absolute"] = "rotary",
    ):
        """
        Protein Masked Language Model.

        Parameters
        ----------
        model_name: pre-trained ESM model (e.g. esm2_t6_8M_UR50D) or name for config (e.g. MLM_small)
        lr: learning rate
        freeze: freeze all layers except LM head (decoder)
        mask_percentage: final masking rate
        initial_mask_percentage: initial masking rate, if not None, linear dynamic mask rate
            scheduler will be used. initial should be greater than final.
        config: huggingfaces config for instantiating a model if ``model_name`` is not specified
        tokenizer_dir: a tokenizer saved to src/transformix/assets
        max_length: max sequence length the model will see

        """
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._freeze = freeze
        self._mask_percentage = mask_percentage
        self._initial_mask_percentage = initial_mask_percentage
        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._max_length = max_length
        self._position_embedding_type = position_embedding_type

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = LMBaseForMaskedLM.from_pretrained(model_name)

        if self._initial_mask_percentage is not None:
            assert self._initial_mask_percentage > self._mask_percentage

        if self._freeze:
            self._freeze_all_but_lm_head()

        self.config = self.model.config
        # if self._continue_training and self._continue_checkpoint is not None:
        #     torch.load(self._continue_checkpoint)
        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        loss, *logging_dicts = self._compute_loss(batch)
        ppl = torch.exp(loss)
        
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/perplexity", ppl, sync_dist=True)
        
        if any(logging_dicts):
            logging_dicts = [{f"train/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        p_mask = self._get_p_mask()
        self.log("train/p_mask", p_mask, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, *logging_dicts = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("valid/loss", loss, sync_dist=True)
        self.log("valid/perplexity", ppl, sync_dist=True)
        if any(logging_dicts):
            logging_dicts = [{f"val/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        return {"valid/loss": loss}

    def _compute_loss(self, batch):
        # torch.cuda.empty_cache()
        tokens = batch["input_ids"].squeeze(1)
        labels = tokens.clone()
        masked_tokens = self._mask_inputs(tokens)
        labels[masked_tokens != self.tokenizer.mask_token_id] = -100  # only calculate loss on masked tokens

        output = self.model(
            input_ids=masked_tokens,
            attention_mask=batch["attention_mask"].squeeze(1),
            labels=labels,
        )
        loss = output["loss"]
        # loss = F.cross_entropy(output["logits"].permute(0, 2, 1), tokens)  # (B, V, L)

        logging_dicts = []

        del masked_tokens, tokens

        return loss, logging_dicts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model for ONNX compiling.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input tensor.
        attention_mask: torch.Tensor
            The attention mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.

        """
        preds = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = preds["hidden_states"]  # hidden reps (B, L, H)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, num layers, L, H)

        return hidden_states

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

    def predict_step(self, batch, batch_idx) -> pd.DataFrame:
        # batch, _targets = batch  # targets are the FASTA IDs
        tokens = batch["input_ids"].squeeze()
        tokens = tokens.to(self.device)
        attention_mask = batch["attention_mask"].squeeze().to(self.device)
        with torch.inference_mode():
            preds = self.model(input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = preds["hidden_states"][-1]  # last layer hidden reps (B, L, H)

        # mean pool over AAs
        df = pd.DataFrame(
            hidden_states.mean(dim=1).cpu(),
            columns=[f"embedding_{idx}" for idx in range(hidden_states.shape[-1])],
        )

        return df

    def _get_p_mask(self):
        if self._initial_mask_percentage is not None:
            p_mask = self._initial_mask_percentage + (self.trainer.global_step / self._num_training_steps) * (
                    self._mask_percentage - self._initial_mask_percentage
            )
        else:
            p_mask = self._mask_percentage

        return p_mask

    def _mask_inputs(self, train_inputs: torch.Tensor, p_mask=None):
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(train_inputs.shape, device=train_inputs.device)
        if p_mask is None:
            p_mask = self._get_p_mask()
        # create mask array
        mask_arr = (
                (rand < p_mask)
                * (train_inputs != self.tokenizer.cls_token_id)
                * (train_inputs != self.tokenizer.pad_token_id)
                * (train_inputs != self.tokenizer.eos_token_id)
        )  # don't mask cls, pad, eos

        selection = []  # masked token positions

        for i in range(train_inputs.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

        masked_inputs = train_inputs.clone()
        for i in range(train_inputs.shape[0]):
            masked_inputs[i, selection[i]] = self.tokenizer.mask_token_id  # 32
        return masked_inputs

    def _freeze_all_but_lm_head(self):
        for name, param in self.model.named_parameters():
            if "lm_head" not in name:  # Skip the lm head
                param.requires_grad = False

    def latent_embeddings_to_sequences(self, x: torch.Tensor) -> list[str]:
        """x: (B, L, H) size tensor of hidden states"""
        with torch.inference_mode():
            logits = self.model.lm_head(x)
        tokens = [self.tokenizer.decode(logit.argmax(dim=-1)) for logit in logits]
        tokens = [t.replace(" ", "") for t in tokens]
       
        return tokens

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self._transform_fn(sequences)])
        with torch.inference_mode():
            hidden_states = self.model(input_ids=input_ids, output_hidden_states=True)["hidden_states"]  # [-1]
        return hidden_states

    def _perturb_seq(self, sequences: list[str], sigma: float = 5.0) -> list[str]:
        h = self.sequences_to_latents(sequences)
        h_perturbed = h + torch.randn(h.shape) * sigma * h.var()
        sequences = self.latent_embeddings_to_sequences(h_perturbed)

        return sequences

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        self.model.save_pretrained(save_directory, *args, **kwargs)
        self.tokenizer.save_pretrained(save_directory, *args, **kwargs)
