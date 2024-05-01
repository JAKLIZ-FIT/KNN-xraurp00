# generate a label file for unanotated data
# which will then be useful for selecting 

# TODO add ability to create label file from generated labels? 

import pandas as pd

save_path = '../bentham_self-supervised/'

label_files= []

label_file = "lines.all"
label_all_df = pd.read_csv(save_path+label_file, sep=" 0 ", header=None, engine='python')
label_all_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
all_cnt = label_all_df.shape[0]
print(f"file: {label_file} number of entries: {all_cnt}")


label_file = "lines.trn"
label_trn_df = pd.read_csv(save_path+label_file, sep=" 0 ", header=None, engine='python')
label_trn_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
trn_cnt = label_trn_df.shape[0]
print(f"file: {label_file} number of entries: {trn_cnt}")

label_file = "lines.val"
label_val_df = pd.read_csv(save_path+label_file, sep=" 0 ", header=None, engine='python')
label_val_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
val_cnt = label_val_df.shape[0]
print(f"file: {label_file} number of entries: {val_cnt}")

label_file = "lines.tst"
label_tst_df = pd.read_csv(save_path+label_file, sep=" 0 ", header=None, engine='python')
label_tst_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
tst_cnt = label_tst_df.shape[0]
print(f"file: {label_file} number of entries: {tst_cnt}")

labeled_cnt = trn_cnt + val_cnt + tst_cnt
print(f"anotated data: {labeled_cnt}   unanotated data: {all_cnt-labeled_cnt}")

filenames_trn = label_trn_df.file_name.to_list()
filenames_val = label_val_df.file_name.to_list()
filenames_tst = label_tst_df.file_name.to_list()

filenames_anotated = filenames_trn + filenames_val + filenames_tst

label_all_df = label_all_df.drop(label_all_df[label_all_df['file_name'].isin(filenames_anotated)].index)
label_all_df['sep'] = 0
label_all_df = label_all_df[['file_name','sep','text']]
print(label_all_df.shape[0])

label_all_df.to_csv(save_path+"lines.unlabeled", sep=" ", header=None)
