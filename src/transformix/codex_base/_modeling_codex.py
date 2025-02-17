from torch import nn
import torch
from transformers.activations import ACT2FN
from transformix import CodexConfig


class CodexProjector(nn.Module):
    def __init__(self, config: CodexConfig):
        super().__init__()
        
        self.protein_model_hidden_size = config.protein_config.hidden_size
        self.text_model_hidden_size = config.text_config.hidden_size
        self.use_bias = config.multimodal_projector_bias
        
        self.linear1 = nn.Linear(
            self.protein_model_hidden_size,
            self.text_model_hidden_size,
            bias=self.use_bias
        )
        
        self.hidden_act = ACT2FN[config.projector_hidden_act]
        
        self.linear2 = nn.Linear(
            self.text_model_hidden_size,
            self.text_model_hidden_size,
            bias=self.use_bias
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.linear1(hidden_states)
        x = self.hidden_act(x)
        
        return self.linear2(x)
