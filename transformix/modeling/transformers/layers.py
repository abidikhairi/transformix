"""Transformers Layers Modules"""
from typing import Optional, Tuple
import torch
from torch import nn

from transformix.modeling.transformers.attention import SelfAttention
from transformix.modeling.transformers.feedforward import OutputLayer, IntermediateLayer


class EncoderLayer(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_attns_heads: int,
                 attns_probs_drop_prob: float = 0.2,
                 output_drop_prob: float = 0.1) -> None:
        super().__init__()

        self.pre_attention_norm = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size=hidden_size, num_attention_heads=num_attns_heads, attention_probs_dropout_prob=attns_probs_drop_prob)
        self.output = OutputLayer(hidden_size=hidden_size, drop_prob=output_drop_prob)
        self.post_attention_norm = nn.LayerNorm(hidden_size)
        self.intermediate = IntermediateLayer(hidden_size=hidden_size, intermediate_size=intermediate_size, drop_prob=output_drop_prob)


    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        residual = hidden_states
        hidden_states = self.pre_attention_norm(hidden_states)
        attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        hidden_states = self.output(attention_outputs[0])

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = hidden_states + residual


        layer_outputs = (hidden_states,) if output_attentions is False else (hidden_states, attention_outputs[1])

        return layer_outputs
