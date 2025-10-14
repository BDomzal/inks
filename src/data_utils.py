import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re

class InksDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index, :], self.y[index, :]
    
    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0]
        return self.X.shape[0]  

def load_training_data(data_path):

    inDKs_df = pd.read_csv(data_path)
    inks_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith('_a')]]
    inks_df.columns = [col.split('_')[0] for col in inks_df.columns]
    inds_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith('_i')]]
    inds_df.columns = [col.split('_')[0] for col in inds_df.columns]
    return inDKs_df, inks_df, inds_df

def create_sample_id(inDKs_df):
    
    inDKs_df = inDKs_df.copy()

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

def remove_outer_samples(inDKs_df, how_many_outer_to_remove):
    
    def remove_outer(group, n):
        if n == 0:
            return group
        else:
            return group.iloc[n:-n] if len(group) > 2*n else pd.DataFrame(columns=group.columns)

    inDKs_df = inDKs_df.groupby('Sample_id', group_keys=False).apply(remove_outer, n=how_many_outer_to_remove)
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
    inDKs_df = inDKs_df.copy()
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    any_negative = (inDKs_df[columns_to_keep_inks + columns_to_keep_inds] < 0).sum().sum() == 0
    if any_negative:
        inDKs_df[columns_to_keep_inks + columns_to_keep_inds] = inDKs_df[columns_to_keep_inks + columns_to_keep_inds].clip(lower=0)
    return inDKs_df

def set_negative_to_zero_v2(inDKs_df, elements_to_keep):
    inDKs_df = inDKs_df.copy()
    any_negative = (inDKs_df[elements_to_keep] < 0).sum().sum() == 0
    if any_negative:
        inDKs_df[elements_to_keep] = inDKs_df[elements_to_keep].clip(lower=0)
    return inDKs_df

def multiply_by_weights(inDKs_df, elements_to_keep, 
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):
    inDKs_df = inDKs_df.copy()
    weights = [el/sum(weights) for el in weights]
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    for i, col in enumerate(columns_to_keep_inds):
        inDKs_df[col] = weights[i]*inDKs_df[col]
    for i, col in enumerate(columns_to_keep_inks):
        inDKs_df[col] = weights[i]*inDKs_df[col]
    return inDKs_df

def multiply_by_weights_v2(inDKs_df, elements_to_keep, 
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):
    inDKs_df = inDKs_df.copy()
    weights = [el/sum(weights) for el in weights]

    for i, col in enumerate(elements_to_keep):
        inDKs_df[col] = weights[i]*inDKs_df[col]

    return inDKs_df

def normalize_to_Fe(inDKs_df, elements_to_keep):
    inDKs_df = inDKs_df.copy()
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    for i, col in enumerate(columns_to_keep_inds):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe_i']
    for i, col in enumerate(columns_to_keep_inks):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe_a']
    inDKs_df = inDKs_df.drop(columns=['Fe_a', 'Fe_i'])
    return inDKs_df

def normalize_to_Fe_v2(inDKs_df, elements_to_keep):
    inDKs_df = inDKs_df.copy()
    for i, col in enumerate(elements_to_keep):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe']
    inDKs_df = inDKs_df.drop(columns=['Fe'])
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
    return device

def data_to_device(X, y, device):
    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)
    return X, y
  
def split_inDKs_df(inDKs_df):
    inds_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith('_i')]].copy()
    inks_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith('_a')]].copy()
    return inds_df, inks_df

def join_to_inDKs_df(inks_df, inds_df, identifiers):
    inDKs_df = pd.concat([inks_df, inds_df, identifiers], axis=1)
    return inDKs_df

def visualise_pca_and_class_means(X_pca, X_sample_ids, 
                                    means_pca, means_sample_ids,
                                    x_lower=None, x_upper=None,
                                    y_lower=None, y_upper=None,
                                    cmap = plt.get_cmap('hsv'),
                                    seed=42, path_to_save=None):

    n_colors = X_sample_ids.nunique()
    colors = [cmap(i / n_colors) for i in range(n_colors)]

    np.random.seed(seed)
    color_dict = dict(zip(np.random.permutation(X_sample_ids.unique()), colors))

    fig, ax = plt.subplots(figsize=(7,7))
    for i, name in enumerate(X_sample_ids):
        ax.plot(X_pca[i, 0], X_pca[i, 1], marker='o', linestyle='', ms=3, color=color_dict[name])
        
    for j, name in enumerate(means_sample_ids):
        ax.plot(means_pca[j, 0], means_pca[j, 1], marker='X', linestyle='', ms=9, color=color_dict[name])
    ax.legend()

    if x_lower is not None and x_upper is not None:
        ax.set_xlim(x_lower, x_upper)
    if y_lower is not None and y_upper is not None:
        ax.set_ylim(y_lower, y_upper)
        
    if path_to_save is not None:
        plt.savefig(path_to_save)

def visualise_train_val_test_distributions(train_data, 
                                            val_data, 
                                            test_data,
                                            elements_to_keep,
                                            title='',
                                            path_to_save=None):

    fig, axes = plt.subplots(len(elements_to_keep)//2, 2, figsize=(7, 7), sharey=False) 

    axes = axes.flatten()

    for i in range(len(elements_to_keep)):

        axes[i].hist(train_data[:,i], bins=100, label='training')
        axes[i].hist(val_data[:,i], bins=100, label='validation')
        axes[i].hist(test_data[:,i], bins=100, label='test')
        axes[i].set_title(title + elements_to_keep[i], x=0.5, y=0.75)

    plt.legend(bbox_to_anchor=(1.6,2.5))

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight')

def load_classes_description_df(classes_description_path):

    classes_df = pd.read_excel(classes_description_path, header=1, usecols=['NAZWA', 'OPIS'])
    classes_df['NAZWA_short'] = classes_df['NAZWA'].apply(lambda x: 
                                                        re.split(r'(\d+)', x)[0] + re.split(r'(\d+)', x)[1] if len(re.split(r'(\d+)', x))>1 else re.split(r'(\d+)', x)[0]
                                                       )                                            
    classes_df.drop_duplicates(['NAZWA', 'OPIS'], inplace=True)
    classes_df_supp = classes_df.drop_duplicates(['NAZWA_short']).copy()
    classes_df_supp['NAZWA'] = classes_df['NAZWA_short']
    classes_df = pd.concat([classes_df, classes_df_supp])
    classes_df.drop_duplicates(inplace=True)
    classes_df.reset_index(inplace=True, drop=True)
    return classes_df


def create_ground_truth_df_for_target_data(which_dataset, 
                                            target_path, 
                                            xrf_path, 
                                            classes_description_df, 
                                            how_many_outer_to_remove=0):

    assert (which_dataset == 'Konstytucja_indicators' or which_dataset == 'Konstytucja_prediction' or which_dataset == 'XRF'), 'Invalid dataset name.'

    def remove_outer(group, n):
        if n == 0:
            return group
        else:
            return group.iloc[n:-n] if len(group) > 2*n else pd.DataFrame(columns=group.columns)

    if which_dataset == 'Konstytucja_indicators' or which_dataset == 'Konstytucja_prediction':
    
        ground_truth_df = pd.read_csv(target_path, usecols=['name'])
        ground_truth_df['Sample_id'] = ground_truth_df['name'].apply(lambda x: x.split('_')[0] if len(x.split('_'))==2 else x.split('_')[0] + x.split('_')[1])
        ground_truth_df['Sample_id'] = ground_truth_df['Sample_id'].apply(lambda x: x.replace('.', ''))
        ground_truth_df['short'] = ground_truth_df['Sample_id'].apply(lambda x: re.split(r'\d+', x)[0])
        ground_truth_df.drop(columns=['name'], inplace=True) 
        ground_truth_df.reset_index(drop=True, inplace=True)
        
        if how_many_outer_to_remove>0:
            ground_truth_df = ground_truth_df.groupby('Sample_id', group_keys=False).apply(remove_outer, n=how_many_outer_to_remove)

    elif which_dataset == 'XRF':

        ground_truth_df = pd.read_csv(xrf_path, usecols=['name'])
        ground_truth_df.rename(columns={'name': 'Sample_id'}, inplace=True)
        ground_truth_df['short'] = ground_truth_df['Sample_id'].apply(lambda x: re.split(r'\d+', x)[0])

    ground_truth_df = pd.merge(left=ground_truth_df, right=classes_description_df, how='left', left_on='Sample_id', right_on='NAZWA')
    
    ground_truth_df['OPIS'] = ground_truth_df['OPIS'].apply(lambda x: str(x).strip())
    ground_truth_df.drop(columns=['NAZWA', 'NAZWA_short'], inplace=True)
    ground_truth_df.reset_index(drop=True, inplace=True)

    return ground_truth_df


def select_book(df, ground_truth_df, book_name):

    assert df.shape[0] == ground_truth_df.shape[0], 'df and ground_truth_df have different lengths!'

    if book_name == 'app':
        df = df[ground_truth_df['short'].apply(lambda x: x.startswith('APP'))].copy()
    elif book_name == 'asc':
        df = df[ground_truth_df['short'].apply(lambda x: x.startswith('ASC'))].copy()
    elif book_name == 'ml':
        df = df[ground_truth_df['short'].apply(lambda x: x.startswith('ML'))].copy()
    elif book_name == 'all':
        pass

    return df


