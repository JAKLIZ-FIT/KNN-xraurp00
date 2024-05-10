# KNN-xraurp00

KNN projekt 2024 - Handwritten text recognition (OCR) - tým xraurp00

### Documentation 
(might get updated)
https://www.overleaf.com/7238591795nmxbwghbjgmm#18e979
  

## Training
Script **trocr_train.py** is used for training. It requires a model, training and validation sets (each split into LMDB file and a file with labels), number of epochs and a destination path where trained model will be stored.  Parameter *-g* signals that a GPU is used for training. Parameter *-b* is for selecting batch size.

    $ python3 trocr_train.py -m models/.trocr-base-stage1 -t /path/to/bentham_self-supervised/lines_40.lmdb -l /path/to/bentham_self-supervised/lines.trn -c /path/to/bentham_self-supervised/lines.val -e 5 -s models/test_stage1_5epochs -g -b 20

  In case the training process is interrupted, it is possible to continue from a checkpoint. When training starts a configuration file is generated with all the training parameters. To resume training using this file specify its location:

    $ python3 trocr_train.py --use-config --config-path /models/test_stage1_5epochs/config.json

## Evaluation
Script **trocr_eval_metrics.py** is used for generating transcripts of images and computing basic performance metrics.

    $ python3 trocr_eval_metrics.py -m models/.trocr-base-stage1 -v /path/to/bentham_self-supervised/lines_40.lmdb -l /path/to/bentham_self-supervised/lines.val -s /save/path/for/results

  Script **eval_metrics.py** is used to compute metrics on the results of the **trocr_eval_metrics.py**. It is used to compute confusion networks as a confidence metric. This script can also select the most confident samples and create a new label file with them. To prevent the selection of mostly the short labels, the script splits the samples into buckets according to the length of their labels.

To compare different confidence metrics:
    
    $ ./eval_metrics.py -d output/confidences_val_aug.csv -o bucket20_metrics.json -m conf_product conf_mean conf_cn -b 20
    
where conf_product, conf_mean, conf_cn are confidence metrics, *-b* parameter is batch size. *-d* is the path to results of **trocr_eval_metrics.py**.

To split data into buckets:

    $ ./eval_metrics.py -d output/confidences_val_aug.csv -b 20 -s conf_cn -p 10 20 30 40 50 60 70 80 90 100 -x cn_
    
where *-p* selects the percentage of data for the output file (i.e. 10 -> 10% of the most confident samples), *-b* is the number of buckets. 
## Data
For information about dataset refer to documentation.
  

### Data loading
Script **extract_ds.py** is used to extract images from the LMDB file.

    $ ./extract_ds.py -s /path/to/lmdb/file -t /target/folder/to/extract/images/into -n 1 --key example.png


### Data augmentation
Script **data_augmentation.py** is used to generate augmented versions of images in the original dataset. Only the images in the label file will be augmented. It generates 8 augmentations variants of each image.

    $ data_augmentation.py --database-path /Path/to/LMDB/database --label-file /Path/to/label_file --output-db-path /Path/to/output/LMDB/database --output-label-file /Path/to/output/label/file

### LMDB tools
 **check_db.py** is used to check for  invalid images in the LMDB database

    $ ./check_db.py -d /path/to/database -l /path/to/label/file
**merge_dbs.py** is used to merge two LMDB databases into one

    $ ./merge_dbs.py -s /path/to/source/lmdb1 /path/to/source/lmdb2 -t /path/to/merged/lmdb

## Other scripts
Other scripts in the repository were used as prototypes.  
