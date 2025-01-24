from typing import Literal

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer
from datasets import Dataset as IndexableDataset, load_dataset as load_huggingface_dataset


class HuggingFaceDataset(TorchDataset):

    def __init__(self, repo_id: str,
                 tokenizer: PreTrainedTokenizer,
                 dataset_split: Literal['train', 'validation', 'test'] = 'train',
                 sequence_max_length: int = 512,
                 sequence_column_name: str = "Sequence"):
        super().__init__()
        self.repo_id: str = repo_id
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self._sequence_max_length: int = sequence_max_length
        self._sequence_column_name: str = sequence_column_name
        self._split: str = dataset_split

        self._dataset: IndexableDataset = load_huggingface_dataset(path=repo_id, split=dataset_split)


    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        row = self._dataset[item]
        sequence = row[self._sequence_column_name]

        inputs = self.tokenizer(
            sequence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self._sequence_max_length
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
