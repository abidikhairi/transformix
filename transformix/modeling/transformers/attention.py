"""Attention Modules"""
import math
from typing import Optional
import torch
from torch import nn

from transformix.modeling.transformers.positional_encoding import RotaryEmbedding


class SelfAttention(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_probs_dropout_prob: float = 0.2):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, False)
        self.key = nn.Linear(hidden_size, self.all_head_size, False)
        self.value = nn.Linear(hidden_size, self.all_head_size, False)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.rotary = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x):
        bs, seq_len, _ = x.shape

        new_x_shape = (bs, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        bs, seq_len, _ = hidden_states.shape
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # [bs, seq_len, num_heads, head_size]      
        query_layer, key_layer = self.rotary(query_layer, key_layer)

              
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.view(bs, seq_len, -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
