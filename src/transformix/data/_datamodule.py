from typing import Literal, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader

from transformix.data import HuggingFaceDataset, LanguageModelingArrowDataset
from transformix.data._arrow_datasets import ProteinTextLanguageModelingArrowDataset


class UnsupervisedHuggingfaceDataModule(pl.LightningDataModule):

    train_data: HuggingFaceDataset
    valid_data: HuggingFaceDataset
    test_data: HuggingFaceDataset

    def __init__(self,
                 repo_id: str,
                 tokenizer: Union[PreTrainedTokenizer, str],
                 batch_size: int = 64,
                 num_proc: int = 4,
                 max_sequence_length: int = 512,
                 sequence_column_name: str = "Sequence"
                 ):
        super().__init__()

        self._repo_id = repo_id

        if isinstance(tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self._tokenizer = tokenizer

        self._batch_size = batch_size
        self._num_proc = num_proc
        self._max_sequence_length = max_sequence_length
        self._sequence_column_name = sequence_column_name

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self._tokenizer,
            return_tensors='pt'
        )

    def setup(self, stage: str) -> None:
        self.train_data = self._load_hf_dataset('train')
        self.valid_data = self._load_hf_dataset('validation')
        self.test_data = self._load_hf_dataset('test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(self.train_data, do_shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(self.valid_data)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(self.test_data)

    def _load_hf_dataset(self, split: Literal['train', 'validation', 'test']):
        return HuggingFaceDataset(
            repo_id=self._repo_id,
            tokenizer=self._tokenizer,
            dataset_split=split,
            sequence_max_length=self._max_sequence_length,
            sequence_column_name=self._sequence_column_name
        )

    def _get_dataloader(self, dataset: HuggingFaceDataset, do_shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=do_shuffle,
            num_workers=self._num_proc,
            collate_fn=self.data_collator
        )

class UnsupervisedArrowCSVDataModule(pl.LightningDataModule):

    train_data: LanguageModelingArrowDataset
    valid_data: LanguageModelingArrowDataset
    test_data: LanguageModelingArrowDataset

    def __init__(
        self,
        base_dir: str,
        tokenizer: Union[PreTrainedTokenizer, str],
        batch_size: int = 64,
        num_proc: int = 4,
        max_sequence_length: int = 512,
        sequence_column_name: str = "Sequence",
        file_ext: str = 'csv',
        sep: str = ',',
        mlm: bool = True
    ):
        super().__init__()

        self._base_dir = base_dir
        self._file_ext = file_ext
        self._sep = sep
        self._mlm = mlm

        if isinstance(tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, padding_side='right')
        else:
            self._tokenizer = tokenizer

        self._batch_size = batch_size
        self._num_proc = num_proc
        self._max_sequence_length = max_sequence_length
        self._sequence_column_name = sequence_column_name

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self._tokenizer,
            return_tensors='pt'
        )

    def setup(self, stage: str) -> None:
        self.train_data = self._load_arrow_dataset('train')
        self.valid_data = self._load_arrow_dataset('validation')
        self.test_data = self._load_arrow_dataset('test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(self.train_data, do_shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(self.valid_data)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(self.test_data)

    def _load_arrow_dataset(self, split: Literal['train', 'validation', 'test']):
        return LanguageModelingArrowDataset(
            base_dir=self._base_dir,
            tokenizer=self._tokenizer,
            dataset_split=split,
            sequence_max_length=self._max_sequence_length,
            sequence_column_name=self._sequence_column_name,
            file_ext=self._file_ext,
            sep=self._sep,
            mlm=self._mlm
        )

    def _get_dataloader(self, dataset: HuggingFaceDataset, do_shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=do_shuffle,
            num_workers=self._num_proc,
            collate_fn=self.data_collator
        )


class PorteinTextLanguageModelingDataModule(pl.LightningDataModule):
    
    train_data: ProteinTextLanguageModelingArrowDataset
    valid_data: ProteinTextLanguageModelingArrowDataset
    test_data: ProteinTextLanguageModelingArrowDataset
    
    def __init__(
        self,
        base_dir: str,
        protein_tokenizer: Union[PreTrainedTokenizer, str],
        text_tokenizer: Union[PreTrainedTokenizer, str],
        max_protein_length: int = 512,
        max_text_length: int = 64,
        target_column_name: str = "Target",
        instruction_column_name: str = "Instruction",
        protein_column_name: str = "Protein",
        file_ext: str = 'csv',
        sep: str = ',',
        batch_size: int = 64,
        num_proc: int = 4,
    ):
        super().__init__()
        
        self._base_dir: str = base_dir
        self._max_protein_length: int = max_protein_length
        self._max_text_length: int = max_text_length
        self._target_column_name: str = target_column_name
        self._instruction_column_name: str = instruction_column_name
        self._protein_column_name: str = protein_column_name
        self._file_ext: str = file_ext
        self._sep: str = sep
        self._batch_size: int = batch_size
        self._num_proc: int = num_proc
        
        if isinstance(protein_tokenizer, str):
            self._protein_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(protein_tokenizer)
        else:
            self._protein_tokenizer: PreTrainedTokenizer = protein_tokenizer
        
        if isinstance(text_tokenizer, str):
            self._text_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        else:
            self._text_tokenizer: PreTrainedTokenizer = text_tokenizer

    
    def setup(self, stage: str = 'train') -> None:
        self.train_data = self._load_arrow_dataset('train')
        self.valid_data = self._load_arrow_dataset('validation')
        self.test_data = self._load_arrow_dataset('test')
    
    def _load_arrow_dataset(self, split: Literal['train', 'validation', 'test']):
        return ProteinTextLanguageModelingArrowDataset(
            base_dir=self._base_dir,
            protein_tokenizer=self._protein_tokenizer,
            text_tokenizer=self._text_tokenizer,
            max_protein_length=self._max_protein_length,
            max_text_length=self._max_text_length,
            dataset_split=split,
            target_column_name=self._target_column_name,
            instruction_column_name=self._instruction_column_name,
            protein_column_name=self._protein_column_name,
            file_ext=self._file_ext,
            sep=self._sep
        )
    
    def _get_dataloader(self, dataset: HuggingFaceDataset, do_shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=do_shuffle,
            num_workers=self._num_proc,
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader(self.train_data, do_shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(self.valid_data)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader(self.test_data)
