# KNN-xraurp00

KNN projekt 2024 - Handwritten text recognition (OCR) - tým xraurp00

  

## Training
Script **trocr_train.py** is used for training. It requires a model, training and validation sets (each split into LMDB file and a file with labels), number of epochs and a destination path where trained model will be stored.  

    $ python3 trocr_train.py -m models/.trocr-base-stage1 -t /path/to/bentham_self-supervised/lines_40.lmdb -l /path/to/bentham_self-supervised/lines.trn -c /path/to/bentham_self-supervised/lines.val -e 5 -s models/test_stage1_5epochs -g

  I case the training process is interrupted, it is possible to continue from a checkpoint. When training starts a configuration file is generated with all the training parameters. To restart the training using this file specify its location:

    $ python3 trocr_train.py --use-config --config-path /models/test_stage1_5epochs/config.json

## Evaluation

  

## Data
Bentham dataset
examples of data

  

### Data loading

  
  

### Data augmentation
