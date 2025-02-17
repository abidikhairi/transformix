from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


class CodexConfig(PretrainedConfig):
    
    model_type = "codex"
    sub_configs = {"text_config": AutoConfig, "protein_config": AutoConfig}
    
    def __init__(
        self,
        protein_config: Optional[PretrainedConfig] = None,
        text_config: Optional[PretrainedConfig] = None,
        ignore_index: int = -100,
        protein_token_id: int = 33,
        projector_hidden_act: str = "gelu",
        protein_seq_length: int = 512,
        multimodal_projector_bias: bool = True,
        **kwargs,
    ):
        """Codex Config

        Args:
            protein_config (Optional[PretrainedConfig], optional): protein lm config. Defaults to None.
            text_config (Optional[PretrainedConfig], optional): language model config. Defaults to None.
            ignore_index (int, optional): index to ignore in cross entropy loss. Defaults to -100.
            protein_token_id (int, optional): special token to represent a protein. Defaults to 33.
            projector_hidden_act (str, optional): hidden activation of the projector. Defaults to "gelu".
            protein_seq_length (int, optional): max protein sequence length. Defaults to 512.
            multimodal_projector_bias (bool, optional): whether to apply bias in multimodal projection. Defaults to True.
        """
        super().__init__(**kwargs, pad_token_id=text_config.pad_token_id)
        
        self.protein_config = protein_config
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.protein_token_id = protein_token_id
        self.projector_hidden_act = projector_hidden_act
        self.protein_seq_length = protein_seq_length
        self.multimodal_projector_bias = multimodal_projector_bias
