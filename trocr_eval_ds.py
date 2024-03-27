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

import pandas as pd

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
    dataloader = DataLoader(ds, 1, shuffle=True, num_workers=8) # batchsize was originally 20
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

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    processor = load_processor()
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

def main():
    args = parse_args()

    output, confidence = predict_for_ds(
        labels_path=Path(args.labels),
        ds_path=Path(args.dataset),
        use_local_model=args.use_local_model
    )

    # TODO - remove
    print(output[:10])
    print(confidence[:10])
    
    for o, c in zip(output, confidence):
        print(f'File: {o[0]}, prediction: {o[1]}, confidence: {c[1]}')
    
    df_output = pd.DataFrame(output, columns =['idx', 'output'])
    df_output['sep1'] = 0
    df_confid = pd.DataFrame(confidence, columns =['idx', 'confidence'])
    df_confid['sep2'] = 0
    df_combi = pd.merge(df_output, df_confid, on='idx', sort=True)
    df_combi = df_combi['idx','sep1','output','sep2','confidence']
    df_combi.to_csv("out.csv",sep=" ")
    
    
    # TODO
    from datasets import load_metric

    cer_metric = load_metric("cer")
    
    
    # calculate CER, WER
    
    
    # put data to file

    return 0

if __name__ == '__main__':
    sys.exit(main())
