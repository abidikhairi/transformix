"""Native Transformer Encoder"""

from typing import Optional
import torch
from torch import nn
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput

from transformix.modeling.masked_lm.config import EncoderBasicConfig
from transformix.modeling.transformers.embedding import Embedding
from transformix.modeling.transformers.layers import EncoderLayer
from transformix.modeling.transformers.positional_encoding import SinusoidalPositionalEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, config: EncoderBasicConfig) -> None:
        super().__init__()

        self.config = config
        
        self.tok_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            do_scale=config.do_scale_embedding,
            token_drop_prob=config.token_drop_prob
        )
        
        self.absolute_pos_embedding = SinusoidalPositionalEmbedding(
            num_positions=config.max_seq_len,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )

        self.encoder = nn.ModuleList([
            EncoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attns_heads=config.num_attention_heads,
                attns_probs_drop_prob=config.attns_probs_drop_prob,
                output_drop_prob=config.output_drop_prob
            )

            for _ in range(config.num_layers)
        ])


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                output_hidden_states: bool = False) -> BaseModelOutput:
        
        bs, seq_len = input_ids.shape

        token_embed = self.tok_embedding(input_ids)
        sinusoidal_pos = self.absolute_pos_embedding(input_ids.size())

        hidden_states = token_embed + sinusoidal_pos

        all_hidden_states = (hidden_states,) if output_hidden_states is True else None
        all_attentions = () if output_attentions is True else None

        for layer in self.encoder:
            layer_outputs = layer(hidden_states, attention_mask, output_attentions=output_attentions)

            if all_hidden_states is not None:
                all_hidden_states += (layer_outputs[0],)
            
            if all_attentions is not None:
                all_attentions += (layer_outputs[1],)

        last_hidden_state = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )