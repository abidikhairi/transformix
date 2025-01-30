from typing import Literal, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader

from transformix.data import HuggingFaceDataset, ArrowCSVDataset


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

    train_data: ArrowCSVDataset
    valid_data: ArrowCSVDataset
    test_data: ArrowCSVDataset

    def __init__(self,
                 base_dir: str,
                 tokenizer: Union[PreTrainedTokenizer, str],
                 batch_size: int = 64,
                 num_proc: int = 4,
                 max_sequence_length: int = 512,
                 sequence_column_name: str = "Sequence",
                 file_ext: str = 'csv',
                 sep: str = ','
                 ):
        super().__init__()

        self._base_dir = base_dir
        self._file_ext = file_ext
        self._sep = sep

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
        return ArrowCSVDataset(
            base_dir=self._base_dir,
            tokenizer=self._tokenizer,
            dataset_split=split,
            sequence_max_length=self._max_sequence_length,
            sequence_column_name=self._sequence_column_name,
            file_ext=self._file_ext,
            sep=self._sep
        )

    def _get_dataloader(self, dataset: HuggingFaceDataset, do_shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=do_shuffle,
            num_workers=self._num_proc,
            collate_fn=self.data_collator
        )
