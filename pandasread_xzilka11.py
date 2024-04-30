#pandasreadmine.py

from evaluate import load

import pandas as pd
df= pd.read_csv('models/base_stage1_augmented_trn/validation_experiment_aug/confidences_val_aug.csv',index_col=0)
#print(df)
#print(len(df['references'].unique()))
df['filenames_short'] = df.filenames.apply(lambda x: x[:19])
df['filenames_end'] = df.filenames.apply(lambda x: x[19:])
#print(df)
#print(df["filenames_short"].unique())

unique_filenames = df["filenames_short"].unique()
unique_GT = df["references"].unique()
unique_augments = df["filenames_end"].unique()

for i, file in enumerate(unique_filenames):
    GT = unique_GT[i]
    print("processing file: "+file+', GT='+GT)
    related_rows = df.loc[df['filenames_short'] == file]
    
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
    related_rows = df.loc[df['filenames_end'] == augment]
    #print(related_rows)
    predictions = related_rows['predictions'].to_list()
    references = related_rows['references'].to_list()

    cer_score = cer.compute(predictions=predictions,references=references)

    print(f"CER score for {augment} files: {cer_score}")
    