from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from dataset import LMDBDataset


@dataclass
class Context:
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor

    train_dataset: LMDBDataset
    train_dataloader: DataLoader

    val_dataset: LMDBDataset
    val_dataloader: DataLoader
