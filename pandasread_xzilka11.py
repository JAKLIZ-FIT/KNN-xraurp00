#!/usr/bin/env python3

#pandasreadmine.py

from evaluate import load

import pandas as pd
import numpy as np
import csv
from pathlib import Path
import sys


def read_labels(label_path:Path) -> pd.DataFrame:
    return pd.read_csv(label_path, sep=" 0 ", header=None, engine='python', quotechar='\\')

def save_labels(in_df : pd.DataFrame, save_path: Path):
    df = in_df.copy()
    df['sep'] = 0

    # select only columns to be added to the label file
    if len(df.references.unique()) == 1:
        df = df[['filenames','sep','predictions']]
    else:
        df = df[['filenames','sep','references']]
    df.rename(columns={0: "filenames", 1: 'sep' ,2: "references"}, inplace=True)

    df.to_csv(save_path,sep=" ",index=False,header=None, quotechar='\\')

def add_unlabeled_to_train(
    eval_results:Path, 
    train_labels:Path, 
    save_path:Path,
    add_amounts:list[int] =[0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    conf_type='Conf_mean',
    generate_new_only=False,
    generate_partial_csv=False,
    generate_augmented_only=False,
    generate_original_only=False,
    generate_extended_train_set=True,
    COMPUTE_CER_WER=False,
    debug_print=False):
    """
    Create label files with added n% most confident MA data

    Go through evaluation results and select most

    :param add_amounts TODO
    """
    
    train_df = read_labels(train_labels)
    train_df.rename(columns={0:'filenames', 1: 'text'}, inplace=True)
    train_files = train_df['filenames'].to_list()
    if debug_print:
        print("train_labels:")
        print(train_df.head())
    
    eval_df= pd.read_csv(eval_results,index_col=0,sep='\\')
    if debug_print:
        print("Eval results:")
        print(eval_df.head())

    # had to adjust for filenames getting longer
    # files include '_' 
    # all files end with "-xxxx.jpg" or "-xxxx_augmenttype.jpg" or "-axxxx.jpg"
    # where x are digits
    eval_df['augment_type'] = eval_df.filenames.apply(lambda x: x.split('-')[-1].split('.')[0])
    if debug_print:
        print("augment types in Eval results:",end=" ")
        print(eval_df['augment_type'].unique())
    eval_df['augment_type'] = eval_df['augment_type'].apply(lambda x: "orig" if len(x.split('_')) == 1 else x.split('_')[1])

    # common part of filename 
    # (for grouping augment variants of the same file)
    eval_df['filenames_short'] = eval_df.filenames.apply(lambda x: "-".join(x.split('-')[:-1])+"-"+(x.split('-')[-1][:4]))

    if debug_print:
        print("extended eval results")
        print(eval_df.head())

    unique_filenames = eval_df["filenames_short"].unique()
    unique_GT = eval_df["references"].unique()
    unique_augments = eval_df["augment_type"].unique()
    
    if debug_print:
        print("augmentation types: ",end="")
        print(unique_augments)
        #print(len(unique_GT))
        #print(unique_GT)
        print(f"number of samples in the evaluated set: {eval_df.shape[0]}")
    
        

    df_augmented_only = None
    df_original_only = None

    if generate_augmented_only or generate_original_only:
        df_augmented_only = eval_df.loc[eval_df['augment_type'] != 'orig']
        df_augmented_only['sep'] = 0
    
        aux = df_augmented_only["augment_type"].unique()
        if debug_print:
            print(f"selecting augmented files: {aux}")
            print(df_augmented_only.head())

        df_original_only = eval_df.loc[eval_df['augment_type'] == 'orig']
        df_original_only['sep'] = 0

        # verify number of samples
        aug_count = df_augmented_only.shape[0]
        print(f"number of augmented samples: {aug_count}")
        orig_count = df_original_only.shape[0]
        print(f"number of original samples: {orig_count}")


    # samples not in training set
    # (subtract train_labels from eval_df)
    new_df = eval_df.loc[~eval_df.filenames.isin(train_files)]
    new_count = new_df.shape[0]
    new_df['sep'] = 0
    gt_count = len(new_df.references.unique())
    if debug_print:
        print(f"number of samples not yet in training set: {new_count}")
        print(f"number of GTs = {gt_count}")
    
    # sort by confidence in descending order
    new_df = new_df.sort_values(conf_type,ascending=False)
    if debug_print:
        print(new_df.head())
        print(new_df.tail())


    for top_size in add_amounts:
        i_top_size = int(top_size*100)
        if debug_print:
            print(f"selecting top {i_top_size}% confident samples by {conf_type}")
        new_filename = "lines"
        file_end = str(i_top_size)+'.trn'
        
        top_count = int(new_count * top_size)
        if debug_print:
            print(f"number of samples={top_count}")
        
        df_top_conf = new_df.iloc[:top_count].copy()
        
        if generate_partial_csv:
            df_top_conf.to_csv(save_path/("lines"+str(i_top_size)+'.csv'),sep="\\") 
            # TODO check the format, but it is useless anyway probably
        
        # label file with only the newly selected MA data
        if generate_new_only: 
            save_labels(df_top_conf,save_path=save_path/(new_filename+"_added"+file_end))

        # label file with original + newly selected
        if generate_extended_train_set:
            df_top_conf = pd.concat([train_df,df_top_conf])
            save_labels(df_top_conf,save_path=save_path/(new_filename+"_extended"+file_end))
        
        # label file with non augmented MA data
        if generate_original_only:
            df_top_conf = df_original_only.iloc[:top_count].copy()
            save_labels(df_top_conf,save_path=save_path/(new_filename+"_aug"+file_end))
        
        # label file with augmented MA data
        if generate_augmented_only:
            df_top_conf = df_augmented_only.iloc[:top_count].copy()
            save_labels(df_top_conf,save_path=save_path/(new_filename+"_orig"+file_end))
            
        # label file with all data
        #if generate_label_file_with_orig:
        #    df_top_plus_orig = pd.concat([df_original_only,df_top_conf])
        #    df_top_plus_orig.to_csv(save_path+new_filename+"_extended"+file_end,sep=" ",index=False,header=None)
        

    compute_cer_per_conf_type = False
    # compute CER scores for augmentations
    if compute_cer_per_conf_type:
        for i, file in enumerate(unique_filenames):
            GT = unique_GT[i]
            print("processing file: "+file+', GT='+GT)
            related_rows = eval_df.loc[eval_df['filenames_short'] == file]
            
            predictions = related_rows['predictions'].to_list()
            print("predictions: ",end="")
            print(predictions)
            confidence_productc = related_rows['Conf_product'].to_list()
            confidence_sum = related_rows['Conf_sum'].to_list()
            # atd

            cer = load("cer")
            cer_score = cer.compute(predictions=predictions,references=[GT]*6)
            car_score = 1 - cer_score
            print(f"CER = {cer_score}")

            cer_score_per_pred = [cer.compute(predictions=[pred],references=[GT]) for pred in predictions]
            print(cer_score_per_pred)
            break


        # compute CER per augmentation
        print(unique_augments)

        for augment in unique_augments:
            related_rows = eval_df.loc[eval_df['augment_type'] == augment]
            #print(related_rows)
            predictions = related_rows['predictions'].to_list()
            references = related_rows['references'].to_list()

            cer_score = cer.compute(predictions=predictions,references=references)

            print(f"CER score for {augment} files: {cer_score}")


def parse_args():
    parser = argparse.ArgumentParser('Create a new label file with X% most confident MA data')
    parser.add_argument(
        '-e', '--eval-results',
        help='Path to the file with results of evaluation Machine-anotated data.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-t', '--train-labels',
        help='Path to file with current training labels.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-a', '--add-percentage',
        help='Percentage of data to be added to train set',
        type=int,
        default = 0
    )
    parser.add_argument(
        '-s', '--save-path',
        help='Path to directory where resulting metrics will be saved.',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-c', '--conf-type',
        help='Confidence meassure to select by.',
        type=str,
        default="Conf_mean"
    )
    parser.add_argument(
        '-p', '--eval-separator',
        help='Separator used in eval file. Old files had \",\".',
        type=str,
        default="\\"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    add_unlabeled_to_train(args.eval_results, args.train_labels, args.save_path, conf_type=args.conf_type,add_amounts=[args.add_percentage])
    

    #10%, 20% 30% 50% 75% a 100%
    #top_sizes = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    
    #eval_results = Path('models/base_stage1_augmented_trn/validation_experiment_aug/confidences_val_aug.csv')
    #eval_results = Path('models/base_stage1_augmented_trn/unlabeled_experiment/checkpoint3.csv')
    #train_labels = Path('../augmented/lines_augmented.trn')
    #save_path = Path('../unlabeled_split/')
    
    #add_unlabeled_to_train(eval_results, train_labels, save_path,debug_print=False)

    return 0

if __name__ == '__main__':
    sys.exit(main())
