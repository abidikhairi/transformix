"""FeedForward Modules"""
import torch
from torch import nn


class IntermediateLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, drop_prob: float = 0.2) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(hidden_size, intermediate_size)

        self.proj_out = nn.Linear(intermediate_size, hidden_size, False)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        
        reset_memory = torch.sigmoid(self.dense1(x))
        x = torch.relu(self.dense2(x))
        
        x = reset_memory * x
        x = self.dropout(x)

        return self.proj_out(x)


class OutputLayer(nn.Module):
    def __init__(self, hidden_size: int, drop_prob: float = 0.1) -> None:
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dense(x)
        x = self.dropout(x)

        return x
