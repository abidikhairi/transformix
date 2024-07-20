"""Embedding Modules"""
import math
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int,
                 do_scale: bool = True,
                 token_drop_prob: float = 0.1
                 ) -> None:
        super().__init__()

        self.do_scale = do_scale
        self.scale_factor = math.sqrt(embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(token_drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        if self.do_scale:
            return x / self.scale_factor
        
        return x
