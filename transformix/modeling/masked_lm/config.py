from pydantic import BaseModel


class EncoderBasicConfig(BaseModel):
    vocab_size: int = 2000
    pad_token_id: int = 0
    hidden_size: int = 128
    intermediate_size: int = 256
    num_attention_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 150
    attns_probs_drop_prob: float = 0.2
    output_drop_prob: float = 0.1
    token_drop_prob: float = 0.1
    do_scale_embedding: bool = True
