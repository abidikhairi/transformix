from transformers import PretrainedConfig, EsmConfig, LlamaConfig


class ProteinTextAdapterConfiguration(PretrainedConfig):
    def __init__(
        self,
        protein_encoder_hidden_size=320,
        text_encoder_hidden_size=576,
        adapter_hidden_size=256,
        adapter_activation='gelu',
        adapter_layer_norm_eps=1e-05,
        protein_token_id=49152,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.protein_encoder_hidden_size = protein_encoder_hidden_size
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.adapter_activation = adapter_activation
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        self.protein_token_id = protein_token_id
        

class ProteinTextLMConfiguration(PretrainedConfig):
    
    model_type = "plm"
    sub_configs = {
        "protein_encoder_config": {},
        "text_encoder_config": {},
        "adapter_config": {}
    }
    
    def __init__(
        self,
        protein_encoder_config=EsmConfig(
            vocab_size=33,
            mask_token_id=32,
            pad_token_id=1,
            is_folding_model=False,
            layer_norm_eps=1e-05,
            hidden_size=320,
            num_hidden_layers=5,
            intermediate_size=1280
        ),
        text_encoder_config=LlamaConfig(
            vocab_size=49153,
            bos_token_id=0,
            eos_token_id=0,
            attention_bias=False,
            is_folding_model=False,
            layer_norm_eps=1e-05,
            hidden_size=576,
            num_hidden_layers=3,
            intermediate_size=1536,
            num_attention_heads=9,
            max_position_embeddings=2048,
            head_dim=64,
            mlp_bias=False
        ),
        adapter_config=ProteinTextAdapterConfiguration(
            protein_encoder_hidden_size=320,
            text_encoder_hidden_size=576,
            adapter_hidden_size=256,
            adapter_activation='gelu',
            adapter_layer_norm_eps=1e-05,
            protein_token_id=49152    
        ),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.protein_encoder_config = protein_encoder_config
        self.text_encoder_config = text_encoder_config
        self.adapter_config = adapter_config        
