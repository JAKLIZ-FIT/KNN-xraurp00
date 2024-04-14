#!/usr/bin/env python3

# LMDB dataset definigion

import lmdb
from pathlib import Path
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

import pandas as pd

def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = dict()
    with open(path, 'r') as file:
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            line = line.split(' ')
            labels[line[0]] = line[2]
    return labels

class LMDBDataset(Dataset):
    def __init__(self, lmdb_database: Path, label_file: Path, processor: TrOCRProcessor, word_len_padding = 8):
        if not lmdb_database.exists():
            raise OSError(f'File {lmdb_database} does not exist!')

        #self.labels = load_labels(label_file)
        label_df = pd.read_csv(label_file, sep=" 0 ", header=None, engine='python')
        label_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
        self.labels = label_df
        self.image_database = lmdb.open(str(lmdb_database))
        self.transaction = self.image_database.begin()
        self.processor = processor
        #self._max_label_len = max([word_len_padding] + [len(self.labels[label]) for label in self.labels.text])
        self._max_label_len = max([word_len_padding] + [self.labels.text.map(len).max()])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        key = self.labels.file_name[idx]
        image = Image.open(BytesIO(self.transaction.get(key.encode()))).convert('RGB')
        image_tensor: torch.tensor = self.processor(image, return_tensors='pt').pixel_values[0]

        label = self.labels.text[idx]
        label_tensor = self.processor.tokenizer(
            label,
            #padding='max_length', truncation=True,# batch_sentences #TODO added  #
            padding = True,
            return_tensors = 'pt',
            pad_to_multiple_of = self._max_label_len
        ).input_ids[0]

        return {'idx': idx, 'input': image_tensor, 'label': label_tensor}
    
    def get_label(self, idx) -> str:
        assert 0 <= idx < len(self.labels.file_name), f"id {idx} outside of bounds [0, {len(self.labels.file_name)}]"
        return self.labels.text[idx]

    def get_path(self, idx) -> str:
        assert 0 <= idx < len(self.labels.file_name), f"id {idx} outside of bounds [0, {len(self.labels.file_name)}]"
        return self.labels.file_name[idx]
