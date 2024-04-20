#!/usr/bin/env python3

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from context import Context
from dataset import LMDBDataset
from evaluate import load
import argparse
import os
import sys
import datetime

from collections import deque
import shutil

# TODO check if necessary
import csv
import pandas as pd

def load_context(
    model_path: Path,
    train_ds_path: Path,
    label_file_path: Path,
    val_label_file_path: Path,
    val_ds_path: Path = None,
    batch_size: int = 20
) -> Context:
    """
    Loads model from local direcotry for training.
    :param model_path (pathlib.Path): path to the model directory
    :param train_ds_path (pathlib.Path): path to training dataset to load
    :param label_file_path (pathlib.Path): path to label file for training dataset
    :param val_label_file_path (pathlib.Path): path to label file for validation dataset
    :param val_ds_path (pathlib.Path): path to validation dataset to load
        (if set to None training dataset is used)
    :param batch_size (int): batch size to use (defaults to 20)
    :return: tuple of lodeded (processor, model)
    """
    for p in (model_path, train_ds_path, label_file_path, val_label_file_path):
        if not p.exists():
            raise ValueError(f'Path {p} does not exist!')
    if val_ds_path and not val_ds_path.exists():
        raise ValueError(f'Path {val_ds_path} does not exist!')
    
    # load and setup model
    processor = TrOCRProcessor.from_pretrained(model_path, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # load ds
    train_ds = LMDBDataset(
        lmdb_database=train_ds_path,
        label_file=label_file_path,
        processor=processor
    )
    if not val_ds_path:
        val_ds_path = train_ds_path
    val_ds = LMDBDataset(
        lmdb_database=val_ds_path,
        label_file=val_label_file_path,
        processor=processor
    )

    # create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    return Context(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        train_dataloader=train_dl,
        val_dataset=val_ds,
        val_dataloader=val_dl
    )

def early_stop_check(last_CAR_scores : deque):
    scores = list(last_CAR_scores) 
    #for score in last_CAR_scores:
    # TODO should I expect the possibility of model getting worse?
    return (scores[-1] - scores[0]) < 0.001 # TODO select a good value

def save_last_scores(last_CAR_scores : deque, save_path : Path):
    with open(save_path+"/scores.txt","w") as f:
            f.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in last_CAR_scores)) 
            #https://stackoverflow.com/questions/3820312/python-write-a-list-of-tuples-to-a-file
    

def train_model(
    context: Context,
    num_epochs: int,
    device: torch.device,
    save_path: Path = ""
):
    """
    Train the provided model.

    Based on code provided by Romeo Sommerfeld
    taken from https://github.com/rsommerfeld/trocr
    file src/scripts.py

    :param context: training context including the model and dataset
    :param num_epochs: number of epochs
    :param device: device to use for training
    """
    model = context.model
    # TODO - use adam from pytorch
    optimizer = AdamW(
        params=model.parameters(),
        lr=5e-6  # TODO - lookup some good values
    )
    num_training_steps = num_epochs * len(context.train_dataloader)
    num_warmup_steps = int(num_training_steps / 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    last_scores_cnt = 20 # TODO add as an argument
    last_CAR_scores = deque(maxlen=last_scores_cnt)
    do_delete_checkpoint = False
    oldest_score = None
    oldest_checkpoint_path = checkpoint_path+"_fail"
    # just some value to generate error if dir does not exist 

    model.to(device)
    model.train()
    timestamp_start = datetime.datetime.now()
    timestamp_last = timestamp_start
    print(f'Training started!\nTime: {timestamp_start}')

    for epoch in range(num_epochs):
        for index, batch in enumerate(context.train_dataloader):
            inputs: torch.Tensor = batch['input'].to(device)
            labels: torch.Tensor = batch['label'].to(device)

            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            del loss, outputs
        
        if len(context.val_dataloader) > 0:
            c_accuracy, w_accuracy = validate(context=context, device=device)
            now = datetime.datetime.now()
            total_time = now - timestamp_start
            epoch_time = now - timestamp_last
            th = int(int(total_time.total_seconds() / 60) / 60)
            tm = int(total_time.total_seconds() / 60) % 60
            ts = int(total_time.total_seconds()) % 60
            eh = int(int(epoch_time.total_seconds() / 60) / 60)
            em = int(epoch_time.total_seconds() / 60) % 60
            es = int(epoch_time.total_seconds()) % 60
            print(f"Epoch: {1 + epoch}")
            print(f"Accuracy: CAR={c_accuracy}, WAR={w_accuracy}")
            print(f"Time: {now}")
            print(f"Total time elapsed: {th}:{tm}:{ts}")
            print(f"Time per epoch: {eh}:{em}:{es}")
            timestamp_last = now

            checkpoint_name = "checkpoint"+str(1+epoch)
            checkpoint_path = save_path + "_" + checkpoint_name
            
            if len(last_CAR_scores == last_scores_cnt): # enough last checkpoints stored
                oldest_score = last_CAR_scores[0]
                oldest_checkpoint_path = save_path+"_"+oldest_score[2]

            last_CAR_scores.append((1+epoch,c_accuracy,checkpoint_name))
            
            if not checkpoint_path.exists(): # save checkpoint
                os.makedirs(checkpoint_path)
            context.processor.save_pretrained(save_directory=checkpoint_path)
            context.model.save_pretrained(save_directory=checkpoint_path)

            # delete oldest checkpoint
            if oldest_checkpoint_path.exists()
                shutil.rmtree(oldest_checkpoint_path)

            if early_stop_check: # early stopping
                save_last_scores(last_CAR_scores,save_path)
                return
    
    # TODO delete checkpoints?
    save_last_scores(last_CAR_scores,save_path)
        

def predict(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    dataloader: DataLoader,
    device: torch.device
) -> tuple[list[tuple[int, str]], list[float]]:
    """
    Predict labels of given dataset.

    Based on code provided by Romeo Sommerfeld
    taken from https://github.com/rsommerfeld/trocr
    file src/scripts.py

    :param processor: processor to use
    :param model: model to use
    :param dataloader: loaded dataset to evaluate
    :returns: (predicted labels, confidence scores)
    """
    output: list[tuple[int, str]] = []
    confidence_scores: list[tuple[int, float]] = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            inputs: torch.Tensor = batch["input"].to(device)

            generated_ids = model.generate(
                inputs=inputs,
                return_dict_in_generate=True,
                output_scores = True
            )
            generated_text = processor.batch_decode(
                generated_ids.sequences,
                skip_special_tokens=True
            )

            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

            # Compute confidence scores
            batch_confidence_scores = get_confidence_scores(
                generated_ids=generated_ids
            )
            confidence_scores.extend(zip(ids, batch_confidence_scores))

    return output, confidence_scores

def get_confidence_scores(generated_ids) -> list[float]:
    """
    Get confidence score for given ids given by probability
    of the predicted sentence.

    Code provided by Romeo Sommerfeld
    taken from https://github.com/rsommerfeld/trocr
    file src/scripts.py

    :param generated_ids: generated predictions
    :returns: list of confidence scores
    """
    # Get raw logits, with shape (examples,tokens,token_vals)
    logits = generated_ids.scores
    logits = torch.stack(list(logits),dim=1)
    print(logits)

    # Transform logits to softmax and keep only the highest
    # (chosen) p for each token
    logit_probs = F.softmax(logits, dim=2)
    char_probs = logit_probs.max(dim=2)[0]

    # Only tokens of val>2 should influence the confidence.
    # Thus, set probabilities to 1 for tokens 0-2
    mask = generated_ids.sequences[:,:-1] > 2
    char_probs[mask] = 1

    # Confidence of each example is cumulative product of token probs
    batch_confidence_scores = char_probs.cumprod(dim=1)[:, -1]
    return [v.item() for v in batch_confidence_scores]

def validate(
    context: Context,
    device: torch.device,
    print_wrong: bool = False
) -> float:
    """
    Validate given model on validation dataset.
    Return the accuracy but doesn't print predictions.

    Based on code provided by Romeo Sommerfeld
    taken from https://github.com/rsommerfeld/trocr
    file src/scripts.py

    :param context: context containing model and datasets
    :param print_wrong: print wrong predictions - defaults to False
    :returns: list of confidence scores
    """
    predictions, _ = predict(
        processor=context.processor,
        model=context.model,
        dataloader=context.val_dataloader,
        device=device
    )
    
    assert len(predictions) > 0
    
    references = [context.val_dataset.get_label(id) for id, prediction in predictions]    
    predictionsList = [prediction for id, prediction in predictions]
    
    cer = load("cer")
    cer_score = cer.compute(predictions=predictions,references=references)
    car_score = 1 - cer_score
    
    wer = load('wer')
    wer_score = wer.compute(predictions=predictions,references=references)
    war_score = 1 - wer_score
    
    return car_score,war_score

def parse_args():
    parser = argparse.ArgumentParser('Train TrOCR model.')
    parser.add_argument(
        '-m', '--model',
        help='Path to the directory with model to use for initialization.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-t', '--training-dataset',
        help='Path to the training dataset directory.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-v', '--validation-dataset',
        help='Path to the validation dataset directory. '
            '(default = same as training dataset)',
        type=Path,
        default=None
    )
    parser.add_argument(
        '-l', '--training-labels',
        help='Path to file with training dataset labels.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-c', '--validation-labels',
        help='Path to file with validation dataset labels.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-e', '--epochs',
        help='Number of epochs to use for training.',
        type=int,
        required=True
    )
    parser.add_argument(
        '-s', '--save-path',
        help='Path to direcotry where resulting trained model will be saved.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-b', '--batch-size',
        help='Batch size to use. (default = 20)',
        type=int,
        default=20
    )
    parser.add_argument(
        '-g', '--use-gpu',
        help='Use GPU (CUDA device) for training instead of CPU. '
            '(default = False)',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '-p', '--num-checkpoints',
        help='Number of checkpoints to store. (default = 20)',
        default=20,
        type=int
    )# TODO pass to train
    return parser.parse_args()

def main():
    args = parse_args()
    # select device
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # load model and dataset
    context = load_context(
        model_path=args.model,
        train_ds_path=args.training_dataset,
        label_file_path=args.training_labels,
        val_label_file_path=args.validation_labels,
        val_ds_path=args.validation_dataset,
        batch_size=args.batch_size
    )
    # train the model
    train_model(context=context, num_epochs=args.epochs, device=device, save_path=args.save_path)
    
    # save results # will be saved as the last checkpoint 
    #if not args.save_path.exists():
    #    os.makedirs(args.save_path)
    #context.processor.save_pretrained(save_directory=args.save_path)
    #context.model.save_pretrained(save_directory=args.save_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
