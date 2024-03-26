#!/usr/bin/env python3
# Evaluate dataset using TrOCR model

import sys
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from trocr.src.util import load_processor, load_model
from trocr.src.scripts import predict
from dataset import LMDBDataset
from context import Context

# TODO
# load val/test ds
# load image db
# for record in ds:
#   load image
#   predict image
#   count CER
#   count WER
#   write result to file
# calculate statistics of CER, WER for all DS
# put data to file

def predict_for_ds(labels_path: Path, ds_path: Path, use_local_model: bool = True):
    processor = load_processor()
    ds = LMDBDataset(
        lmdb_database=ds_path,
        label_file=labels_path,
        processor=processor
    )
    dataloader = DataLoader(ds, 20, shuffle=True, num_workers=8)
    model = load_model(use_local_model)

    """
    context = Context(
        model=model,
        processor=processor,
        train_dataset=None,
        train_dataloader=None,
        val_dataset=ds,
        val_dataloader=dataloader
    )
    """

    return predict(
        processor=processor,
        model=model,
        dataloader=dataloader
    )

def parse_args():
    parser = argparse.ArgumentParser('Evaluate dataset.')
    parser.add_argument(
        '-l', '--labels',
        help='Path to file with labels.',
        required=True
    )
    parser.add_argument(
        '-d', '--dataset',
        help='Path to file with dataset (lmdb).',
        required=True
    )
    parser.add_argument(
        '-m', '--use-local-model',
        help='Use local model or download one from huggingface.',
        default=False,
        action='store_true'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    output, confidence = predict_for_ds(
        labels_path=Path(args.labels),
        ds_path=Path(args.dataset),
        use_local_model=args.use_local_model
    )

    # TODO - remove
    for o, c in zip(output, confidence):
        print(f'File: {o[0]}, prediction: {o[1]}, confidence: {c[1]}')

    # TODO
    # calculate CER, WER
    # put data to file

    return 0

if __name__ == '__main__':
    sys.exit(main())
