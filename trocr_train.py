#!/usr/bin/env python3

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
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
    val_ds_path: Path = None
) -> Context:
    """
    Loads model from local direcotry for training.
    :param model_path (pathlib.Path): path to the model directory
    :param train_ds_path (pathlib.Path): path to training dataset to load
    :param label_file_path (pathlib.Path): path to label file for training dataset
    :param val_label_file_path (pathlib.Path): path to label file for validation dataset
    :param val_ds_path (pathlib.Path): path to validation dataset to load
        (if set to None training dataset is used)
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
    train_dl = DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=20, num_workers=0)

    return Context(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        train_dataloader=train_dl,
        val_dataset=val_ds,
        val_dataloader=val_dl
    )

def train_model(
    context: Context,
    num_epochs: int,
    device: torch.device
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
    optimizer = AdamW(
        params=model.parameters(),
        lr=1e-6
    )
    num_training_steps = num_epochs * len(context.train_dataloader)
    num_warmup_steps = int(num_training_steps / 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

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
            accuracy = validate(context=context, device=device)
            now = datetime.datetime.now()
            total_time = now - timestamp_start
            epoch_time = now - timestamp_last
            th = int(int(total_time / 60) / 60)
            tm = int(total_time / 60) % 60
            ts = total_time % 60
            eh = int(int(epoch_time / 60) / 60)
            em = int(epoch_time / 60) % 60
            es = epoch_time % 60
            print(f"Epoch: {1 + epoch}")
            print(f"Accuracy: {accuracy}")
            print(f"Time: {now}")
            print(f"Total time elapsed: {th}:{tm}:{ts}")
            print(f"Time per epoch: {eh}:{em}:{es}")
            timestamp_last = now

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
            debug_print(f"Predicting batch {i+1}")
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
        train_ds_path=args.training_dataset,
        label_file_path=args.training_labels,
        val_label_file_path=args.validation_labels,
        val_ds_path=args.validation_dataset
    )
    # train the model
    train_model(context=context, num_epochs=args.epochs, device=device)
    # save results
    if not args.save_path.exist():
        os.makedirs(args.save_path)
    context.processor.save_pretrained(save_directory=args.save_path)
    context.model.save_pretrained(save_directory=args.save_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
