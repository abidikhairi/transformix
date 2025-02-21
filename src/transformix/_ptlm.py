from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.activations import ACT2FN

from transformix._ptlm_configuration import (
    ProteinTextLMConfiguration,
    ProteinTextAdapterConfiguration
)


class ProteinAdapter(nn.Module):
    def __init__(
        self,
        adapter_config: ProteinTextAdapterConfiguration
    ):
        super().__init__()
        
        self.config = adapter_config
        
        self.linear1 = nn.Linear(self.config.protein_encoder_hidden_size, self.config.adapter_hidden_size)
        self.activation = ACT2FN[self.config.adapter_activation]
        self.linear2 = nn.Linear(self.config.adapter_hidden_size, self.config.text_encoder_hidden_size)
        
        self.layer_norm = nn.LayerNorm(self.config.text_encoder_hidden_size, eps=self.config.adapter_layer_norm_eps)

    def forward(self, hidden_states):
        x = self.activation(self.linear1(hidden_states))
        x = self.linear2(x)
        x = self.layer_norm(x)
        
        return x


class ProteinTextLanguageModel(PreTrainedModel):
    model_tags = ["protein-text-lm", "plm", "multimodal-lm"]
    
    config_class = ProteinTextLMConfiguration
    
    def __init__(
        self, 
        config: Optional[PretrainedConfig] = None,
        protein_encoder: Optional[Union[PreTrainedModel, str]] = None,
        text_encoder: Optional[Union[PreTrainedModel, str]] = None,
        adapter: Optional[Union[ProteinAdapter, ProteinTextAdapterConfiguration, Dict[str, Any]]] = None
    ):
        if config is None and (protein_encoder is None or text_encoder is None):
            raise ValueError("Either config or protein_encoder, text_encoder and adapter should be provided")
        
        if config is None:
            config = ProteinTextLMConfiguration()
            if protein_encoder is not None:
                if isinstance(protein_encoder, str):
                    protein_encoder = AutoModel.from_pretrained(protein_encoder)
                else:
                    protein_encoder = protein_encoder
                    
            else:
                raise ValueError("protein_encoder should be provided, when config is None")
            
            if text_encoder is not None:
                if isinstance(text_encoder, str):
                    text_encoder = AutoModelForCausalLM.from_pretrained(text_encoder)
                else:
                    text_encoder = text_encoder
                    
            else:
                raise ValueError("text_encoder should be provided, when config is None")
        
            if adapter is not None:
                if isinstance(adapter, ProteinTextAdapterConfiguration):
                    adapter = ProteinAdapter(adapter)
                elif isinstance(adapter, Dict):
                    adapter = ProteinAdapter(ProteinTextAdapterConfiguration.from_dict(adapter))
                else:
                    adapter = adapter
            else:
                # Create adapter with default config
                adapter = ProteinAdapter(ProteinTextAdapterConfiguration())
            
            config.protein_encoder_config = protein_encoder.config
            config.adapter_config = adapter.config
            config.text_encoder_config = text_encoder.config
        else:
            protein_encoder = AutoModel.from_config(config.protein_encoder_config)
            text_encoder = AutoModelForCausalLM.from_config(config.text_encoder_config)
            adapter = ProteinAdapter(config.adapter_config)
        
        super().__init__(config)

        self.protein_encoder = protein_encoder
        self.text_encoder = text_encoder
        self.adapter = adapter

 
    def _generate_4d_attention_mask(
        self,
        protein_attention_mask: torch.Tensor,
        text_attention_mask: torch.Tensor
    ):
        
        assert protein_attention_mask.device == text_attention_mask.device
        
        device = protein_attention_mask.device
        
        batch_size = text_attention_mask.size(0)
        protein_seq_len = protein_attention_mask.size(1)
        text_seq_len = text_attention_mask.size(1)
        num_heads = self.config.text_encoder_config.num_attention_heads
        pad = (0, protein_seq_len - text_seq_len, 0, 0)
        
        original_text_attention_mask = text_attention_mask.clone()
        original_protein_attention_mask = protein_attention_mask.clone()

        # create 4D mask for protein
        # [bs, protein_seq_len, protein_seq_len] -> 1 attend to and 0 do not attend
        protein_attention_mask = protein_attention_mask[:, None ,:] * protein_attention_mask[:, :, None]
        left_upper_block = protein_attention_mask

        # [batch_size, protein_seq_len, text_seq_len]
        right_upper_block = torch.zeros((batch_size, protein_seq_len, text_seq_len)).to(device)
        # left_upper_block[:, :, 0] = 1 # first text token is <protein> it should attend to protein tokens
        # TODO(khairi): temporary remove, needs to be discussed
       
        # create 4D causal mask for text
        # [bs, seq_len, seq_len] -> 1 attend to and 0 do not attend
        causal_mask = torch.tril(torch.ones(text_seq_len, text_seq_len)).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        text_attention_mask = text_attention_mask[:, None, :] * text_attention_mask[:, :, None]
        text_attention_mask = causal_mask & text_attention_mask
        
        right_lower_block = torch.nn.functional.pad(text_attention_mask, pad)
        right_lower_block = text_attention_mask
        
        left_lower_block = torch.ones((batch_size, text_seq_len, protein_seq_len)).to(device)
        left_lower_block = left_lower_block * original_text_attention_mask[:, :, None]
        left_lower_block = left_lower_block * original_protein_attention_mask[:, None, :]
        
        upper_block = torch.cat([left_upper_block, right_upper_block], dim=2)
        lower_block = torch.cat([left_lower_block, right_lower_block], dim=2)
        
        # concat both        
        attention_mask = torch.cat([upper_block, lower_block], dim=1)

        # 0 = attend to, -inf = do not attend
        attention_mask = (1 - attention_mask).float() * torch.finfo(torch.float32).min
        
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
        # attention_mask = attention_mask.to(self.device)
        
        return attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_input_ids: Optional[torch.Tensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
    ):
        protein_outputs = self.protein_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask
        )

        protein_embeds = protein_outputs.last_hidden_state # use all tokens as protein level representation
        
        # apply adapter
        protein_embeds = self.adapter(protein_embeds)
        
        # generate only text embeddings at this step
        text_embeds = self.text_encoder.model.embed_tokens(input_ids)

        # concat protein and text embeddings
        inputs_embeds = torch.cat([protein_embeds, text_embeds], dim=1)
                
        # generate custom attention mask
        attention_mask = self._generate_4d_attention_mask(
            protein_attention_mask,
            attention_mask
        )

        # generate text
        text_output = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        return text_output


class TransformixProteinTextLanguageModel(pl.LightningModule):
    def __init__(
        self, 
        config: Optional[ProteinTextLMConfiguration] = None,
        protein_encoder: Optional[Union[str, PreTrainedModel]] = None,
        text_encoder: Optional[Union[str, PreTrainedModel]] = None,
        adapter: Optional[Union[ProteinTextAdapterConfiguration, ProteinAdapter, Dict[str, Any]]] = None,
        learning_rate: float = 1e-5,
        beta1: float = 0.99,
        beta2: float = 0.98,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        freeze_protein_encoder: bool = True,
        freeze_text_encoder: bool = True,
        freeze_adapter: bool = False
    ):
        super().__init__()
        
        self.model = ProteinTextLanguageModel(
            config=config,
            protein_encoder=protein_encoder,
            text_encoder=text_encoder,
            adapter=adapter
        )
        
        if freeze_protein_encoder:
            self.model.protein_encoder.requires_grad_(False)
        
        if freeze_text_encoder:
            self.model.text_encoder.requires_grad_(False)
        
        if freeze_adapter:
            self.model.adapter.requires_grad_(False)
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        self.config = self.model.config
        self.save_hyperparameters()
        
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        protein_input_ids = batch["protein_input_ids"]
        protein_attention_mask = batch["protein_attention_mask"]
        
        protein_len = protein_input_ids.shape[1]
        vocab_size = self.config.text_encoder_config.vocab_size
        
        labels = batch["input_ids"][:, 1:].clone()
        labels[attention_mask == 0] = -100
        
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask
        )
        
        logits = outputs.logits[:, protein_len:, :].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )
        
        # sync_dist=True: synchronize logging between devices (in case of multi-gpu training)
        self.log('train/loss', loss)
        self.log('train/perplexity', loss.exp())
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        protein_input_ids = batch["protein_input_ids"]
        protein_attention_mask = batch["protein_attention_mask"]
        
        protein_len = protein_input_ids.shape[1]
        vocab_size = self.config.text_encoder_config.vocab_size
        
        labels = batch["input_ids"][:, 1:].clone()
        labels[attention_mask == 0] = -100
        
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask
        )
        
        logits = outputs.logits[:, protein_len:, :].contiguous()

        loss = self.loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )
        
        # sync_dist=True: synchronize logging between devices (in case of multi-gpu training)
        self.log('valid/loss', loss, sync_dist=True)
        self.log('valid/perplexity', loss.exp(), sync_dist=True)
        
        return {"loss": loss}

    
    def predict_step(
        self,
        sequences: Union[str, List[str]],
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
    ):
        raise NotImplementedError("Predict step is not implemented")
     
    def push_to_hub(self, repo_id):
        self.model.push_to_hub(repo_id)
