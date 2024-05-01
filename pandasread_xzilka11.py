#pandasreadmine.py

from evaluate import load

import pandas as pd

#10%, 20% 30% 50% 75% a 100%
top_sizes = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
# select confidence measure
conf_type='Conf_mean'

generate_partial_csv = False # generate partial csvs similar to input csv 
generate_label_file_only_new = False # label file with only the newly selected
generate_label_file_with_orig = False # label file with original + newly selected

compute_cer_per_conf_type = False

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
df_original_only = df.loc[df['filenames_end'] == '.jpg']
aux = df_augmented_only["filenames_end"].unique()
print(f"selecting augmented files: {aux}")
#print(df_augmented_only.head())

    
# sort by confidence in descending order
df_augmented_only = df_augmented_only.sort_values(conf_type,ascending=False)

# verify number of samples
aug_count = df_augmented_only.shape[0]
print(f"number of augmented samples: {aug_count}")

df_original_only_copy = df_original_only.copy()
df_original_only_copy['sep'] = 0
df_original_only_copy = df_original_only_copy[['filenames','sep','references']]
df_original_only_copy.rename(columns={0: "filenames", 1: 'sep' ,2: "predictions"}, inplace=True)
# predictions just to match column name later

for top_size in top_sizes:
    i_top_size = int(top_size*100)
    print(f"selecting top {i_top_size}% confident samples by {conf_type}")
    
    
    top_count = int(aug_count * top_size)
    print(f"number of samples={top_count}")
    df_top_conf = df_augmented_only.iloc[:top_count]
    
    if generate_partial_csv:
        df_top_conf.to_csv(save_path+load_filename+str(i_top_size)+'.csv')
        
    # label file with only the newly selected
    df_top_conf['sep'] = 0
    df_top_conf = df_top_conf[['filenames','sep','predictions']]
    if generate_label_file_only_new:
        df_top_conf.to_csv(save_path+lines_new+str(i_top_size)+'.trn')
        
    # label file with original + newly selected
    if generate_label_file_with_orig:
        df_top_plus_orig = pd.concat([df_original_only,df_top_conf])
        df_top_plus_orig.to_csv(save_path+lines_new+str(i_top_size)+'.trn')
    


# compute CER scores for augmentations
if compute_cer_per_conf_type:
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