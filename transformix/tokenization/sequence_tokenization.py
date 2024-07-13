"Sequence tokenizer"
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from transformix.data import constants


class SequenceTokenizer(PreTrainedTokenizerFast):
    """
    Constructs a sequence tokenizer
    """

    def __init__(self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        all_tokens = constants.AMINO_ACIDS

        token_to_id = {tok: id for id, tok in enumerate(all_tokens)}

        tokenizer = Tokenizer(
            BPE(vocab=token_to_id, merges=[], unk_token=unk_token)
        )

        special_tokens = [cls_token, pad_token, mask_token, eos_token]

        tokenizer.add_special_tokens(
            special_tokens,
        )

        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls>:0 $A:0 <eos>:0",
            pair="<cls>:0 $A:0 <eos>:0 $B:1 <eos>:1",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id
