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
import argparse
import os
import sys
import datetime

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
    for p in (model_path, val_label_file_path):
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
    if not val_ds_path:
        val_ds_path = train_ds_path
    val_ds = LMDBDataset(
        lmdb_database=val_ds_path,
        label_file=val_label_file_path,
        processor=processor
    )

    # create dataloaders
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    return Context(
        model=model,
        processor=processor,
        train_dataset=None,
        train_dataloader=None,
        val_dataset=val_ds,
        val_dataloader=val_dl
    )

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

    model.to(device)

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
            print(generated_text)
            exit(0)
    return output, confidence_scores

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

    """
    predictions, confidence_scores = predict(
        processor=context.processor,
        model=context.model,
        dataloader=context.val_dataloader,
        device=device
    )
    
    assert len(predictions) > 0

    correct_count = 0
    wrong_count = 0
    for id, prediction in predictions:
        label = context.val_dataset.get_label(id)
        path = context.val_dataset.get_path(id)

        # TODO - Use CER, WER, not whole label?
        if prediction == label:
            correct_count += 1
        else:
            wrong_count += 1
            if print_wrong:
                print(f"Predicted: \t{prediction}\nLabel: \t\t{label}\nPath: \t\t{path}")

    if print_wrong:
        print(f"\nCorrect: {correct_count}\nWrong: {wrong_count}")
    return correct_count / (len(predictions))
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

def get_confidence_scores(generated_ids) -> list[float]:
    """
    Get confidence score for given ids given by probability
    of the predicted sentence.

    Code provided by Romeo Sommerfeld
    taken from https://github.com/rsommerfeld/trocr
    file src/scripts.py
a
    :param generated_ids: generated predictions
    :returns: list of confidence scores
    """
    # Get raw logits, with shape (examples,tokens,token_vals)
    logits = generated_ids.scores
    #print(logits)
    logits = torch.stack(list(logits),dim=1)
    #print(logits)

    # Transform logits to softmax and keep only the highest
    # (chosen) p for each token
    logit_probs = F.softmax(logits, dim=2)
    char_probs = logit_probs.max(dim=2)[0]
    print(char_probs)
    char_probs_for_mean = torch.clone(char_probs)
    
    # Only tokens of val>2 should influence the confidence.
    # Thus, set probabilities to 1 for tokens 0-2
    #mask = generated_ids.sequences[:,:-1] > 2 # original implementation
    mask = generated_ids.sequences[:,:-1] <= 2 # TODO is this correct ???
    print(mask)
    mask4 = generated_ids.sequences[:,:-1] < 2
    print(mask4)
    mask2 = torch.clone(mask)
    mask3 = generated_ids.sequences[:,:-1] > 2
    valid_char_count = mask3.cumsum(dim=1)[:, -1]
    print(valid_char_count)

    char_probs[mask] = 1
    print("\nCharProbsMasked:\n")
    print(char_probs)
    char_probs_for_mean[mask] = 0
    
    # Confidence of each example is cumulative product of token probs
    batch_confidence_scores = char_probs.cumprod(dim=1)[:, -1]
    batch_confidence_scores2 = char_probs_for_mean.cumsum(dim=1)[:, -1]
    batch_confidence_scores3 = torch.max(char_probs_for_mean,dim=1)
    batch_confidence_scores4 = torch.min(char_probs_for_mean,dim=1)
    print("cumprod")
    print(batch_confidence_scores)
    print("cumsum")
    print(batch_confidence_scores2)
    batch_confidence_scores2 = torch.div(batch_confidence_scores2,valid_char_count)
    print("mean")
    print(batch_confidence_scores2)
    print("max:")
    print(batch_confidence_scores3)
    print("min:")
    print(batch_confidence_scores4)
    # TODO change return
    return [v.item() for v in batch_confidence_scores], 


def parse_args():
    parser = argparse.ArgumentParser('Train TrOCR model.')
    parser.add_argument(
        '-m', '--model',
        help='Path to the directory with model to use for initialization.',
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
        '-l', '--validation-labels',
        help='Path to file with validation dataset labels.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-s', '--save-path',
        help='Path to directory where resulting metrics will be saved.',
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
    return parser.parse_args()

def main():
    args = parse_args()
    # select device
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # load model and dataset
    context = load_context(
        model_path=args.model,
        train_ds_path=None,
        label_file_path=None,
        val_label_file_path=args.validation_labels,
        val_ds_path=args.validation_dataset,
        batch_size=args.batch_size
    )

    # TODO add logits and save logits
    accuracy = validate(context=context, device=device)
    # save results
    if not args.save_path.exists():
        os.makedirs(args.save_path)
    # TODO save logits to file?
    return 0


if __name__ == '__main__':
    sys.exit(main())