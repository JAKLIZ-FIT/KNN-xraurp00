#pandasreadmine.py

from evaluate import load

import pandas as pd
#save_path = 'models/base_stage1_augmented_trn/validation_experiment_aug/'
save_path = 'models/base_stage1_augmented_trn/train_set_eval_experiment_aug/'
#load_filename = 'confidences_val_aug'
load_filename = 'confidences_trn_aug'
df= pd.read_csv(save_path+load_filename+'.csv',index_col=0)
#print(df)
#print(len(df['references'].unique()))
df['filenames_short'] = df.filenames.apply(lambda x: x[:19])
df['filenames_end'] = df.filenames.apply(lambda x: x[19:])
#print(df)
#print(df["filenames_short"].unique())

unique_filenames = df["filenames_short"].unique()
unique_GT = df["references"].unique()
unique_augments = df["filenames_end"].unique()

print(f"number of samples in the whole set{df.shape[0]}")
df_augmented_only = df.loc[df['filenames_end'] != '.jpg']
aux = df_augmented_only["filenames_end"].unique()
print(f"selecting augmented files: {aux}")

#print(df_augmented_only.head())

"""
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
"""
top_size = 0.3
#10%, 20% 30% 50% 75% a 100%
top_sizes = [0.1,0.2,0.3,0.5,0.75,1.0]
conf_type='Conf_mean'
df_augmented_only = df_augmented_only.sort_values(conf_type,ascending=False)
aug_count = df_augmented_only.shape[0]
print(f"number of augmented samples: {aug_count}")

for top_size in top_sizes:
    i_top_size = int(top_size*100)
    print(f"selecting top {i_top_size}% confident samples by {conf_type}")
    
    top_count = int(aug_count * top_size)
    df_top_conf = df_augmented_only.iloc[:top_count]
    df_top_conf = df_top_conf[['ids','references','predictions','Conf_product', 'Conf_sum', 'Conf_max', 'Conf_mean', 'Conf_min','filenames']]
    df_top_conf.to_csv(save_path+load_filename+str(i_top_size)+'.csv')
    print(f"number of samples={top_count}")