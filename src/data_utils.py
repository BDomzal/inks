import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

def load_training_data(data_path):

    df = pd.read_csv(data_path)
    inks_df = df[[col for col in df.columns if col.endswith('_a')]]
    inks_df.columns = [col.split('_')[0] for col in inks_df.columns]
    inds_df = df[[col for col in df.columns if col.endswith('_i')]]
    inds_df.columns = [col.split('_')[0] for col in inds_df.columns]
    return inks_df, inds_df

def create_sample_id(inDKs_df):

    def extract_base_name(name):
        return name.split('_')[0].split('.')[0]

    assert all(
        inDKs_df['name_i'].apply(extract_base_name) ==
        inDKs_df['name_a'].apply(extract_base_name)
    )
    
    inDKs_df['Sample_id'] = inDKs_df['name_i'].apply(extract_base_name)
    inDKs_df['name'] = (
        inDKs_df['Sample_id'] + '_' +
        inDKs_df['name_i'].apply(lambda x: x.split('_')[-1])
    )
    return inDKs_df

def remove_outer_samples(inDKs_df, nr_of_samples):
    
    def remove_outer(group, n):
        return group.iloc[n:-n] if len(group) > 2*n else pd.DataFrame(columns=group.columns)

    inDKs_df = inDKs_df.groupby('Sample_id', group_keys=False).apply(remove_outer, n=nr_of_samples)
    inDKs_df = inDKs_df.reset_index(drop=True)
    return inDKs_df	

def delete_elements(inDKs_df, elements_to_keep):
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    inDKs_df = inDKs_df[columns_to_keep_inks + columns_to_keep_inds + ['Sample_id', 'name']]
    return inDKs_df

def remove_missing_data(inDKs_df):
    how_many_nans = (inDKs_df.shape[0] - inDKs_df.dropna().shape[0]) / inDKs_df.shape[0]
    if how_many_nans > 0:
        inDKs_df = inDKs_df.dropna()
    return inDKs_df

def set_negative_to_zero(inDKs_df, elements_to_keep):
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    any_negative = (inDKs_df[columns_to_keep_inks + columns_to_keep_inds] < 0).sum().sum() == 0
    if any_negative:
        inDKs_df[columns_to_keep_inks + columns_to_keep_inds] = inDKs_df[columns_to_keep_inks + columns_to_keep_inds].clip(lower=0)
    return inDKs_df

def multiply_by_weights(inDKs_df, elements_to_keep, 
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    for i, col in enumerate(columns_to_keep_inds):
        inDKs_df[col] = weights[i]*inDKs_df[col]
    for i, col in enumerate(columns_to_keep_inks):
        inDKs_df[col] = weights[i]*inDKs_df[col]
    return inDKs_df

def normalize_to_Fe(inDKs_df, elements_to_keep):
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    for i, col in enumerate(columns_to_keep_inds):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe_i']
    for i, col in enumerate(columns_to_keep_inks):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe_a']
    inDKs_df = inDKs_df.drop(columns=['Fe_a', 'Fe_i'])
    return inDKs_df

def create_partition(inDKs_df):
    indices = list(range(int(inDKs_df.shape[0])))
    ind_train_all, ind_test_all = train_test_split(indices, test_size=0.2, random_state=1)
    ind_train_all, ind_val_all = train_test_split(ind_train_all, test_size=0.25, random_state=1)

    partition = {'train': ind_train_all,
            'val': ind_val_all,
            'test': ind_test_all}

    X_y_train = inDKs_df.iloc[partition['train'],:]
    X_y_val = inDKs_df.iloc[partition['val'],:]
    X_y_test = inDKs_df.iloc[partition['test'],:]

    X_y_train.reset_index(drop=True, inplace=True)
    X_y_val.reset_index(drop=True, inplace=True)
    X_y_test.reset_index(drop=True, inplace=True)
    return X_y_train, X_y_val, X_y_test

def prepare_data_for_training(X_y_train, X_y_val, X_y_test, elements_to_keep):
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    train_order = X_y_train['Sample_id']
    val_order = X_y_val['Sample_id']
    test_order = X_y_test['Sample_id']

    X_train = np.array(X_y_train[columns_to_keep_inds].values)
    y_train = np.array(X_y_train[columns_to_keep_inks].values)
    X_val = np.array(X_y_val[columns_to_keep_inds].values)
    y_val = np.array(X_y_val[columns_to_keep_inks].values)
    X_test = np.array(X_y_test[columns_to_keep_inds].values)
    y_test = np.array(X_y_test[columns_to_keep_inks].values)
    return X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order

def transform_data(X, y, preprocessing_method):
    
    def adjusted_log_transform(input_array):
        res = np.where(input_array>0, np.log(input_array), 0.)
        return res
    
    if preprocessing_method == 'normalization':

        X = (X - np.min(X, axis=0))/np.std(X, axis=0)
        y = (y - np.min(y, axis=0))/np.std(y, axis=0)
    
    elif preprocessing_method == 'logarithm':
        
        X = adjusted_log_transform(X)
        y = adjusted_log_transform(y)

    elif preprocessing_method == 'logarithm_and_normalization':
        
        #logarithm
        X = adjusted_log_transform(X)
        y = adjusted_log_transform(y)
        
        #normalization
        X = (X - np.min(X, axis=0))/np.std(X, axis=0)
        y = (y - np.min(y, axis=0))/np.std(y, axis=0)

    return X, y

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

def data_to_device(X, y, device):
    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)
    return X, y

class InksDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index, :], self.y[index, :]
    
    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0]
        return self.X.shape[0]    
    
