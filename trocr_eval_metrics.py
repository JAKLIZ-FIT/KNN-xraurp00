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

import pickle
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
    device: torch.device,
    save_path: Path
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


    char_probs_list: list[torch.Tensor] = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            inputs: torch.Tensor = batch["input"].to(device)

            generated_ids = model.generate(
                inputs=inputs,
                return_dict_in_generate=True,
                output_scores = True,
                max_length = 40
            )
            generated_text = processor.batch_decode(
                generated_ids.sequences,
                skip_special_tokens=True
            )

            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

            # Compute confidence scores
            batch_confidence_scores, char_probs = get_confidence_scores(
                generated_ids=generated_ids, save_path=save_path
            )

            char_probs_list.append(char_probs)
            #print(type(char_probs),end="\t")
            #print(char_probs.dim(),end="\t")
            #print(char_probs.size(dim=0),end="\t")
            #print(char_probs.size(dim=1))
            #confidence_scores.extend(zip(ids, batch_confidence_scores))
            confidence_scores.extend(zip(ids, batch_confidence_scores[0], batch_confidence_scores[1], batch_confidence_scores[2], batch_confidence_scores[3],batch_confidence_scores[4]))
            #print(generated_text)
            #print(confidence_scores)
            
    #char_probs = torch.cat(char_probs_list)
    #rint(type(char_probs))
    #print(char_probs.dim())
    #print(char_probs.size(dim=0),end=" ")
    #print(char_probs.size(dim=1))

    #if not save_path.exists():
    #    os.makedirs(save_path)
    #torch.save(char_probs,save_path/"char_probs.pt")
    
    return output, confidence_scores

def validate(
    context: Context,
    device: torch.device,
    save_path: Path,
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
    predictions, confidences = predict(
        processor=context.processor,
        model=context.model,
        dataloader=context.val_dataloader,
        device=device,
        save_path=save_path

    )
    
    assert len(predictions) > 0

    # TODO save CER and Confidence together for selection of data to add to trainDS
    # TODO calculate CER per label
    # TODO plot results
    # TODO integrate (area under curve) to get the best confidence metric
    
    references = [context.val_dataset.get_label(id) for id, prediction in predictions]    
    predictionsList = [prediction for id, prediction in predictions]
    ids = [id for id, prediction in predictions]
    filenames = [context.val_dataset.get_path(id) for id, prediction in predictions]

    cer = load("cer")
    cer_score = cer.compute(predictions=predictionsList,references=references)
    car_score = 1 - cer_score
    
    wer = load('wer')
    wer_score = wer.compute(predictions=predictionsList,references=references)
    war_score = 1 - wer_score


    results_df = pd.DataFrame(confidences, columns =['ids','Conf_product', 'Conf_sum', 'Conf_max', 'Conf_mean', 'Conf_min'])
    print(results_df.head())
    #results_df['ids'] = ids
    results_df['references'] = references
    results_df['predictions'] = predictionsList
    results_df['filenames'] = filenames
    print(results_df.columns)
    results_df = results_df[['ids','references','predictions','Conf_product', 'Conf_sum', 'Conf_max', 'Conf_mean', 'Conf_min','filenames']]
    print(results_df)
    print(results_df.sort_values('ids'))

    results_df.to_csv(save_path/'confidences_val_aug.csv')
    print(car_score)
    print(war_score)
    return car_score,war_score

def get_confidence_scores(generated_ids, save_path:Path) -> list[float]:
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
    #print("logit tensor dimensions=",end="") 
    #aux = [logitsa.dim() for logitsa in logits]
    #print(len(aux))
    #print(aux)
    #aux2 = [(logitsa.size(dim=0),logitsa.size(dim=0)) for logitsa in logits]
    #print(len(aux2))
    #print(aux2)

    #print(logits)
    logits = torch.stack(list(logits),dim=1)
    print(type(logits),end="\t")
    print(logits.dim(),end="\t")
    print(logits.size(dim=0),end="\t")
    print(logits.size(dim=1),end="\t")
    print(logits.size(dim=2),end="\t")

    #if not save_path.exists():
    #    os.makedirs(save_path)
    #torch.save(logits,save_path/"logits.pt")
    

    # Transform logits to softmax and keep only the highest
    # (chosen) p for each token
    logit_probs = F.softmax(logits, dim=2)
    print(type(logit_probs),end="\t")
    print(logit_probs.dim(),end="\t")
    print(logit_probs.size(dim=0),end="\t")
    print(logit_probs.size(dim=1),end="\t")
    print(logit_probs.size(dim=2))
    char_probs = logit_probs.max(dim=2)[0]
    char_probs_for_mean = torch.clone(char_probs)
    char_probs_clean = torch.clone(char_probs)

    #if not save_path.exists():
    #    os.makedirs(save_path)
    #torch.save(char_probs,save_path/"char_probs.pt")
    
    # Only tokens of val>2 should influence the confidence.
    # Thus, set probabilities to 1 for tokens 0-2
    #mask = generated_ids.sequences[:,:-1] > 2 # original implementation
    mask = generated_ids.sequences[:,:-1] <= 2 # TODO is this correct ???
    mask_inverted = generated_ids.sequences[:,:-1] > 2
    valid_char_count = mask_inverted.cumsum(dim=1)[:, -1]
    char_probs[mask] = 1
    char_probs_for_mean[mask] = 0
    
    # Confidence of each example is cumulative product of token probs
    batch_confidence_scores_prod = char_probs.cumprod(dim=1)[:, -1]
    batch_confidence_scores_sum = char_probs_for_mean.cumsum(dim=1)[:, -1]
    batch_confidence_scores_mean = torch.div(batch_confidence_scores_sum,valid_char_count)
    batch_confidence_scores_max = torch.max(char_probs_for_mean,dim=1)[0]
    batch_confidence_scores_min = torch.min(char_probs_for_mean,dim=1)[0]
    #print("cumprod")
    #print(batch_confidence_scores_prod)
    #print("cumsum")
    #print(batch_confidence_scores_mean)
    #batch_confidence_scores_mean = torch.div(batch_confidence_scores_mean,valid_char_count)
    #print("mean")
    #print(batch_confidence_scores_mean)
    #print("max:")
    #print(batch_confidence_scores_max)
    #print("min:")
    #print(batch_confidence_scores_min)
    # TODO change return
    return [[v.item() for v in batch_confidence_scores_prod], \
           [v.item() for v in batch_confidence_scores_sum],\
           [v.item() for v in batch_confidence_scores_max],\
           [v.item() for v in batch_confidence_scores_mean],\
           [v.item() for v in batch_confidence_scores_min]],\
           char_probs_clean


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

    # TODO add confusion network
    # TODO add logits and save logits
    accuracy = validate(context=context, device=device, save_path=args.save_path)
    # save results
    if not args.save_path.exists():
        os.makedirs(args.save_path)
    # TODO save logits to file?
    return 0


if __name__ == '__main__':
    sys.exit(main())