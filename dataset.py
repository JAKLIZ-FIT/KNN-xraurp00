#!/usr/bin/env python3

# LMDB dataset definigion

import lmdb
from pathlib import Path
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

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

        self.labels = load_labels(label_file)
        self.image_database = lmdb.open(str(lmdb_database))
        self.transaction = self.image_database.begin()
        self.processor = processor
        self._max_label_len = max([word_len_padding] + [len(self.labels[label]) for label in self.labels])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = Image.open(BytesIO(self.transaction.get(idx))).convert('RGB')
        image_tensor: torch.tensor = self.processor(image, return_tensors='pt').pixel_values[0]

        label = self.labels[idx]
        label_tensor = self.processor.tokenizer(
            label,
            return_tensors = 'pt',
            padding = True,
            pad_to_multiple_of = self._max_label_len
        ).input_ids[0]

        return {'idx': idx, 'input': image_tensor, 'label': label_tensor}
