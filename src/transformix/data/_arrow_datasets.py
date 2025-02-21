import os.path as osp
from typing import Literal

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer
from datasets import Dataset as IndexableDataset


class LanguageModelingArrowDataset(TorchDataset):

    def __init__(self, 
                 base_dir: str,
                 tokenizer: PreTrainedTokenizer,
                 dataset_split: Literal['train', 'validation', 'test'] = 'train',
                 sequence_max_length: int = 512,
                 sequence_column_name: str = "Sequence",
                 file_ext: str = 'csv',
                 sep: str = ',',
                 mlm: bool = True
                 ):
        super().__init__()
        self.base_dir: str = base_dir
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self._sequence_max_length: int = sequence_max_length
        self._sequence_column_name: str = sequence_column_name
        self._split: str = dataset_split

        self._dataset: IndexableDataset = IndexableDataset.from_csv(
            path_or_paths=osp.join(self.base_dir, f'{self._split}.{file_ext}'),
            sep=sep 
        )
        
        if not mlm:    
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

class ProteinTextLanguageModelingArrowDataset(TorchDataset):

    def __init__(
        self, 
        base_dir: str,
        protein_tokenizer: PreTrainedTokenizer,
        text_tokenizer: PreTrainedTokenizer,
        max_protein_length: int = 512,
        max_text_length: int = 64,
        dataset_split: Literal['train', 'validation', 'test'] = 'train',
        target_column_name: str = "Target",
        instruction_column_name: str = "Instruction",
        protein_column_name: str = "Protein",
        file_ext: str = 'csv',
        sep: str = ',',
    ):
        super().__init__()
        
        self.base_dir: str = base_dir
        self.protein_tokenizer: PreTrainedTokenizer = protein_tokenizer
        self.text_tokenizer: PreTrainedTokenizer = text_tokenizer
        self._max_protein_length: int = max_protein_length
        self._max_text_length: int = max_text_length
        self._target_column_name: str = target_column_name
        self._instruction_column_name: str = instruction_column_name
        self._protein_column_name: str = protein_column_name
        self._split: str = dataset_split

        self._dataset: IndexableDataset = IndexableDataset.from_csv(
            path_or_paths=osp.join(self.base_dir, f'{self._split}.{file_ext}'),
            sep=sep 
        )
        
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token        


    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        row = self._dataset[item]
        
        sequence = row[self._protein_column_name]
        instruction = row[self._instruction_column_name]
        target = row[self._target_column_name]
        
        text = f'<protein> {instruction}: {target}'
        
        text_inputs = self.text_tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self._max_text_length
        )
        
        protein_inputs = self.protein_tokenizer(
            sequence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self._max_protein_length
        )

        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        protein_input_ids = protein_inputs['input_ids'].squeeze(0)
        protein_attention_mask = protein_inputs['attention_mask'].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "protein_input_ids": protein_input_ids,
            "protein_attention_mask": protein_attention_mask
        }