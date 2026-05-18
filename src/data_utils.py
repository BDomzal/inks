import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
import math
from collections import Counter
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

def load_training_data(data_path, indicators_suffix='_i', inks_suffix='_a', standardise_names=True):

    inDKs_df = pd.read_csv(data_path)
    inks_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith(inks_suffix)]]
    inds_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith(indicators_suffix)]]

    if standardise_names:
        inks_df.columns = [col.split('_')[0] for col in inks_df.columns]
        inds_df.columns = [col.split('_')[0] for col in inds_df.columns]

    return inDKs_df, inks_df, inds_df

def load_target_data(target_path, header=0):
    inds_df = pd.read_csv(target_path, header=header)
    return inds_df

def split_inDKs_df(inDKs_df, indicators_suffix='_i', inks_suffix='_a', standardise_names=False):

    inds_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith(indicators_suffix)]].copy()
    inks_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith(inks_suffix)]].copy()

    if standardise_names:
        inks_df.columns = [col.replace(inks_suffix, '') for col in inks_df.columns]
        inds_df.columns = [col.replace(indicators_suffix, '') for col in inds_df.columns]

    return inds_df, inks_df

def join_to_inDKs_df(inks_df, inds_df, identifiers):

    inDKs_df = pd.concat([inks_df, inds_df, identifiers], axis=1)
    return inDKs_df

def create_sample_id_in_training_data(inDKs_df, name_indicators='name_i', name_inks='name_a'):
    
    inDKs_df = inDKs_df.copy()

    def extract_base_name(name):
        return name.split('_')[0].split('.')[0]

    assert all(
        inDKs_df[name_indicators].apply(extract_base_name) ==
        inDKs_df[name_inks].apply(extract_base_name)
    )
    
    inDKs_df['Sample_id'] = inDKs_df[name_indicators].apply(extract_base_name)
    inDKs_df['name'] = (
        inDKs_df['Sample_id'] + '_' +
        inDKs_df[name_indicators].apply(lambda x: x.split('_')[-1])
    )

    return inDKs_df

def create_sample_id_in_target_data(inds_df, column_to_use='name'):

    assert column_to_use in inds_df.columns

    inds_df['Sample_id'] = inds_df[column_to_use].apply(lambda x: x.split('_')[0] if len(x.split('_'))==2 else x.split('_')[0] + x.split('_')[1])
    #inds_df['Sample_id'] = inds_df['Sample_id'].apply(lambda x: x.split('.')[0])
    inds_df.drop(columns=[column_to_use], inplace=True)
    inds_df.reset_index(drop=True, inplace=True)

    return inds_df

def remove_outer_samples(any_df, how_many_outer_to_remove, sample_id_column='Sample_id'):
    
    def remove_outer(group, n):
        if n == 0:
            return group
        else:
            return group.iloc[n:-n] if len(group) > 2*n else pd.DataFrame(columns=group.columns)

    any_df = any_df.groupby(sample_id_column, group_keys=False).apply(remove_outer, n=how_many_outer_to_remove)
    any_df = any_df.reset_index(drop=True)

    return any_df

def delete_elements(any_df, elements_to_keep, indicators_suffix='_i', inks_suffix='_a', keep_sample_id=True, keep_name=True):

    columns_to_keep_inds = [el + indicators_suffix for el in elements_to_keep]
    columns_to_keep_inks = [el + inks_suffix for el in elements_to_keep]

    if keep_sample_id:
        elements_to_keep = elements_to_keep + ['Sample_id']
    if keep_name:
        elements_to_keep = elements_to_keep + ['name']

    return any_df[[col for col in any_df.columns if col in elements_to_keep + columns_to_keep_inds + columns_to_keep_inks]]

def remove_missing_data(any_df):

    how_many_nans = any_df.shape[0] - any_df.dropna().shape[0]
    if how_many_nans > 0:
        any_df = any_df.dropna()

    return any_df

def set_negative_to_zero(any_df):

    cols = any_df.select_dtypes(np.number).columns
    any_df[cols] = any_df[cols].clip(lower=0)

    return any_df

def multiply_by_weights(any_df, columns_to_transform, column_suffix='',
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):

    any_df = any_df.copy()
    weights = [el/sum(weights) for el in weights]
    columns_to_transform = [el + column_suffix for el in columns_to_transform]
    for i, col in enumerate(columns_to_transform):
        any_df[col] = weights[i]*any_df[col]

    return any_df

def divide_by_weights(any_df, columns_to_transform, suffix='',
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):

    any_df = any_df.copy()
    weights = [el/sum(weights) for el in weights]
    columns_to_transform = [el + suffix for el in columns_to_transform]
    for i, col in enumerate(columns_to_transform):
        any_df[col] = any_df[col]/weights[i]

    return any_df

def normalise_to_Fe(any_df, elements_to_keep, remove_Fe=False, suffixes=['', '_i', '_a']):

    any_df = any_df.copy()

    for suffix in suffixes:
        divisor = any_df['Fe' + suffix].copy()
        columns_to_keep = [el + suffix for el in elements_to_keep]
        for col in columns_to_keep:
            any_df[col] = any_df[col] / divisor

    if remove_Fe:
        any_df = any_df.drop(columns=['Fe' + suffix for suffix in suffixes])

    return any_df

def normalise_to_total(df):
    df = np.exp(df)/(np.exp(df).sum(1)).values.reshape(-1,1)
    return df

def truncate_names(y_true):
    y_true = y_true.apply(lambda x: re.split(r'(\d+|\.|UR)', x)[0] if x.startswith('Konstytucja') else x.split('_')[0])
    return y_true


def create_partition(inDKs_df, random_state=3):
    indices = list(range(int(inDKs_df.shape[0])))
    ind_train_all, ind_test_all = train_test_split(indices, test_size=0.2, random_state=random_state)
    ind_train_all, ind_val_all = train_test_split(ind_train_all, test_size=0.25, random_state=random_state)

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

def split_to_X_and_y(X_y_train, X_y_val, X_y_test, elements_to_keep, indicators_suffix='_i', inks_suffix='_a'):
    columns_to_keep_inds = [el + indicators_suffix for el in elements_to_keep]
    columns_to_keep_inks = [el + inks_suffix for el in elements_to_keep]
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

def transform_data(X, preprocessing_method):
    
    def adjusted_log_transform(input_array):
        res = np.where(input_array>0, np.log(input_array), 0.)
        return res
    
    if preprocessing_method == 'normalisation':

        X = (X - np.min(X, axis=0))/np.std(X, axis=0)
    
    elif preprocessing_method == 'logarithm':
        
        X = adjusted_log_transform(X)

    elif preprocessing_method == 'logarithm_and_normalisation':
        
        #logarithm
        X = adjusted_log_transform(X)
        
        #normalisation
        X = (X - np.min(X, axis=0))/np.std(X, axis=0)

    return X

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

def data_to_device(X, device):
    X = torch.Tensor(X).to(device)
    return X

def preprocessing_beginning(
                            data_path,
                            how_many_outer_to_remove,
                            elements_to_keep,
                            multiplication_weights,
                            indicators_suffix='_i', 
                            inks_suffix='_a',
                            normalisation_to_Fe=False
                            ):

    inDKs_df, inks_df, inds_df = load_training_data(data_path, indicators_suffix=indicators_suffix, inks_suffix=inks_suffix)

    # ## Preprocessing

    # 0. Keeping track of the records from the same sample.
    # We will keep this info in 'Sample_id' and 'name' columns.

    inDKs_df = create_sample_id_in_training_data(inDKs_df)

    # 1. Removing 'outer' samples:

    inDKs_df = remove_outer_samples(inDKs_df, how_many_outer_to_remove)


    # 2. Removing columns that we don't need.
    # Instead of predicting amounts of all the elements, we will predict only those from elements_to_keep list.

    inDKs_df = delete_elements(inDKs_df, elements_to_keep)

    # 3. Removing rows with missing values if there are any.

    inDKs_df = remove_missing_data(inDKs_df)

    # 4. Setting negative numbers to zeros. (First, checking if there are any.)

    inDKs_df = set_negative_to_zero(inDKs_df)

    # 5. Dividing the indicators by weights (leaving inks as they are!)

    inDKs_df = divide_by_weights(inDKs_df, elements_to_keep, suffix=indicators_suffix, weights=multiplication_weights)


    if normalisation_to_Fe:

        # 6. Normalising with respect to Fe.

        inDKs_df = normalise_to_Fe(inDKs_df, elements_to_keep, suffixes=[indicators_suffix, inks_suffix])

        # 7. Removing Fe.

        elements_to_keep_no_fe = [el for el in elements_to_keep if el != 'Fe']
        inDKs_df = delete_elements(inDKs_df, elements_to_keep_no_fe)

    return inDKs_df

def prepare_training_data(
                        data_path,
                        how_many_outer_to_remove,
                        elements_to_keep,
                        multiplication_weights,
                        preprocessing_method, 
                        indicators_suffix='_i', 
                        inks_suffix='_a',
                        random_state=3,
                        normalisation_to_Fe=False,
                        return_data=True,
                        perturb_data=False,
                        mu=0,
                        sigma=0.1,
                        return_original_labels=False
                        ):

    inDKs_df = preprocessing_beginning(
                                        data_path,
                                        how_many_outer_to_remove,
                                        elements_to_keep,
                                        multiplication_weights,
                                        indicators_suffix=indicators_suffix, 
                                        inks_suffix=inks_suffix,
                                        normalisation_to_Fe=normalisation_to_Fe
                                        )
    

    # 8. Train - val - test split.

    X_y_train, X_y_val, X_y_test = create_partition(inDKs_df, random_state=random_state)


    # 9. Creating features and labels matrices.

    elements_to_keep_no_fe = [el for el in elements_to_keep if el != 'Fe']

    if return_original_labels:
        original_labels = X_y_test['name']

    X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = split_to_X_and_y(
                                                                                                            X_y_train, 
                                                                                                            X_y_val, 
                                                                                                            X_y_test, 
                                                                                                            elements_to_keep_no_fe if normalisation_to_Fe else elements_to_keep)


    # 10. Normalisation / taking logarithm.
    # (It is done after splitting because normalisation takes into account info from every sample in the input dataset.
    # Therefore, normalisation of val and test data must be done separately, in order not to use information from train data.)

    X_train = transform_data(X_train, preprocessing_method)
    y_train = transform_data(y_train, preprocessing_method)
    X_val = transform_data(X_val, preprocessing_method)
    y_val = transform_data(y_val, preprocessing_method)
    X_test = transform_data(X_test, preprocessing_method)
    y_test = transform_data(y_test, preprocessing_method)


    # 10 1/2. Optional perturbation of X in test_data.
    if perturb_data:
        X_test = X_test + np.random.normal(mu, sigma, X_test.shape[0]).reshape(-1, 1)


    # 11. Converting to tensors.
    if return_data:
        data_to_return = [X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order]

    device = get_device()

    X_train = data_to_device(X_train, device)
    y_train = data_to_device(y_train, device)
    X_val = data_to_device(X_val, device)
    y_val = data_to_device(y_val, device)
    X_test = data_to_device(X_test, device)
    y_test = data_to_device(y_test, device)


    # 12. Creating InksDatasets.

    train_dataset = InksDataset(X=X_train, y=y_train)
    val_dataset = InksDataset(X=X_val, y=y_val)
    test_dataset = InksDataset(X=X_test, y=y_test)

    # 13. Creating DataLoaders.

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=100)
    val_loader = DataLoader(val_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    if return_data:
        if return_original_labels:
            return train_loader, val_loader, test_loader, data_to_return, original_labels
        else:
            return train_loader, val_loader, test_loader, data_to_return
    else:
        if return_original_labels:
            return train_loader, val_loader, test_loader, original_labels
        else:
            return train_loader, val_loader, test_loader


def prepare_data_without_splitting(
                                    data_path,
                                    how_many_outer_to_remove,
                                    elements_to_keep,
                                    multiplication_weights,
                                    preprocessing_method,
                                    indicators_suffix='_i', 
                                    inks_suffix='_a',
                                    normalisation_to_Fe=False
                                    ):

    inDKs_df = preprocessing_beginning(
                                        data_path,
                                        how_many_outer_to_remove,
                                        elements_to_keep,
                                        multiplication_weights,
                                        indicators_suffix=indicators_suffix, 
                                        inks_suffix=inks_suffix,
                                        normalisation_to_Fe=normalisation_to_Fe
                                        )
    
    elements_to_keep_no_fe = [el for el in elements_to_keep if el != 'Fe']

    # 8. Creating features and labels matrices.

    if normalisation_to_Fe:
        columns_to_keep_inds = [el + indicators_suffix for el in elements_to_keep_no_fe]
        columns_to_keep_inks = [el + inks_suffix for el in elements_to_keep_no_fe]
    else:
        columns_to_keep_inds = [el + indicators_suffix for el in elements_to_keep]
        columns_to_keep_inks = [el + inks_suffix for el in elements_to_keep]

    sample_order = inDKs_df['Sample_id']

    X = np.array(inDKs_df[columns_to_keep_inds].values)
    y = np.array(inDKs_df[columns_to_keep_inks].values)

    # 9. Normalisation / taking logarithm.

    X = transform_data(X, preprocessing_method)
    y = transform_data(y, preprocessing_method)

    inDKs_df[columns_to_keep_inds] = X
    inDKs_df[columns_to_keep_inks] = y

    inds_df, inks_df = split_inDKs_df(inDKs_df)
    inds_df.columns = [name.split('_')[0] for name in inds_df.columns]
    inks_df.columns = [name.split('_')[0] for name in inks_df.columns]
    inds_df['name'] = inDKs_df['name']
    inks_df['name'] = inDKs_df['name']

    return inds_df, inks_df, sample_order

def sample_in_list(sample, list_of_names):
    return any([sample.startswith(el) for el in list_of_names])


def visualise_pca_and_class_means(X_pca, 
                                    X_sample_ids, 
                                    means_pca, 
                                    means_sample_ids,
                                    x_lower=None, 
                                    x_upper=None,
                                    y_lower=None, 
                                    y_upper=None,
                                    cmap = plt.get_cmap('hsv'),
                                    seed=42, 
                                    dimensions=[0,1],
                                    hide_ticks=False,
                                    path_to_save=None):

    n_colors = X_sample_ids.nunique()

    dim0, dim1 = dimensions

    #continuous cmap
    colors = [cmap(i / n_colors) for i in range(n_colors)]

    #discrete cmap with 20 colors
    #colors = [cmap(i%20) for i in range(n_colors)] 

    np.random.seed(seed)
    color_dict = dict(zip(np.random.permutation(X_sample_ids.unique()), colors))

    fig, ax = plt.subplots(figsize=(7,7))
    for i, name in enumerate(X_sample_ids):
        ax.plot(X_pca[i, dim0], X_pca[i, dim1], marker='o', linestyle='', ms=8, color=color_dict[name])
        
    for j, name in enumerate(means_sample_ids):
        ax.plot(means_pca[j, dim0], means_pca[j, dim1], marker='X', linestyle='', ms=20, color=color_dict[name]) #marker's colorful filling
        ax.plot(means_pca[j, dim0], means_pca[j, dim1], marker='X', linestyle='', ms=20, color='black', fillstyle='none') #marker's black frame

    ax.legend()

    plt.xlabel('PC'+str(dim0+1), fontsize=20)
    plt.ylabel('PC'+str(dim1+1), fontsize=20)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if x_lower is not None and x_upper is not None:
        ax.set_xlim(x_lower, x_upper)
    if y_lower is not None and y_upper is not None:
        ax.set_ylim(y_lower, y_upper)

    plt.tight_layout()
        
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)
    plt.show()


def visualise_train_val_test_distributions(train_data, 
                                            val_data, 
                                            test_data,
                                            elements_to_keep,
                                            title='',
                                            horizontal_axis_name='',
                                            logarithmed=True,
                                            lower_x_lim=None,
                                            upper_x_lim=None,
                                            lower_y_lim=None,
                                            upper_y_lim=None,
                                            path_to_save=None):

    fig, axes = plt.subplots(math.ceil(len(elements_to_keep)/2), 2, figsize=(7, 7), sharey=True, sharex=True) 

    axes = axes.flatten()

    if not logarithmed:

        train_data = np.exp(train_data)
        val_data = np.exp(val_data)
        test_data = np.exp(test_data)

    for i in range(len(elements_to_keep)):

        axes[i].hist(train_data[:,i], bins=80, label='training', color='#007C91')
        axes[i].hist(val_data[:,i], bins=80, label='validation', color='#E66100')
        axes[i].hist(test_data[:,i], bins=80, label='test', color='#4B4B4B')
        axes[i].set_title(elements_to_keep[i], x=0.5, y=0.75)
        if lower_x_lim is not None and upper_x_lim is not None:
            axes[i].set_xlim([lower_x_lim, upper_x_lim])
        if lower_y_lim is not None and upper_y_lim is not None:
            axes[i].set_ylim([lower_y_lim, upper_y_lim])

    fig.suptitle(title, fontsize=25)
    fig.supxlabel(horizontal_axis_name)
    plt.legend(bbox_to_anchor=(1.6,2.5))

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', dpi=300)
    plt.show()

def save_prediction(prediction, results_path, model_name):
    outputs_to_save = prediction.cpu().detach().numpy()
    np.savetxt(results_path + 'prediction_for_new_dataset_from_' + model_name, outputs_to_save, delimiter=',')

def to_numpy(pytorch_tensor):
    return(pytorch_tensor.cpu().detach().numpy())


def load_prediction(prediction_path, elements_to_keep):
        
    input_data = np.loadtxt(prediction_path, delimiter=',')
    df = pd.DataFrame(data=input_data, columns=elements_to_keep)
    
    return df

def visualise_pca(X_low_dim, y,
                    dimensions = [0,1],
                    figures_name = 'pca',
                    cmap=plt.get_cmap('tab20'),
                    annotate = False,
                    whether_sort = True,
                    figures_path=None):

    colors = cmap(np.linspace(0, 0.99, y.nunique()))

    if whether_sort:
        sorted_labels = sorted(y.unique())
    else:
        sorted_labels = y.unique()

    color_dict = dict((key, value) for key, value in zip(sorted_labels, colors))
    color_dict['corroded'] = [1., 0., 0., 1.]
    legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor=color_dict[label],
                                                  label=label, markersize=15) for label in sorted_labels]
    
    x_pca_0 = [X_low_dim[i, dimensions[0]] for i in range(X_low_dim.shape[0])]
    x_pca_1 = [X_low_dim[i, dimensions[1]] for i in range(X_low_dim.shape[0])]
    y_colors = np.array([color_dict[el] for el in y])

    plt.xlabel('PC' + str(dimensions[0]+1))
    plt.ylabel('PC' + str(dimensions[1]+1))

    plt.scatter(x_pca_0, x_pca_1, marker='o', s=100, color=y_colors)
    plt.legend(handles=legend_elements, prop={'size': 8})
            
    if figures_path is not None:
        plt.savefig(figures_path + '_' + figures_name + '_' + 'PC' + str(dimensions[0]+1) + '_' + 'PC' + str(dimensions[1]+1) + '.png')
    plt.show()

def groupby_and_get_mean_of_first_n(df, y_true, n=5):

    df['Label'] = y_true
    #df = df.groupby('Label').agg('mean')
    df = df.groupby('Label').agg(lambda x: x[:n].mean())

    y_true = pd.Series(df.index)
    df.reset_index(inplace=True, drop=True)

    return df, y_true

def translate_names(current_names, official_names_dict):
    return current_names.apply(lambda x: official_names_dict[x])

def visualise_selected_axis(
                            X, y,
                            elements_to_keep,
                            dimensions_horizontal = [1, 4, 7],
                            dimensions_vertical = [8, 9, 10],
                            figsize=(7, 5),
                            cmap=plt.get_cmap('tab20'),
                            annotate = False,
                            whether_sort = True,
                            path_to_save=None):

    fig, ax = plt.subplots(figsize=figsize)

    colors = cmap(np.linspace(0, 0.99, y.nunique()))

    if whether_sort:
        sorted_labels = sorted(y.unique())
    else:
        sorted_labels = y.unique()

    color_dict = dict((key, value) for key, value in zip(sorted_labels, colors))
    color_dict['corroded'] = [1., 0., 0., 1.]
    legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor=color_dict[label],
                                                  label=label, markersize=15) for label in sorted_labels]
    
    x_h = [sum([X[i, j] for j in dimensions_horizontal]) for i in range(X.shape[0])]
    x_v = [sum([X[i, j] for j in dimensions_vertical]) for i in range(X.shape[0])]
    y_colors = np.array([color_dict[el] for el in y])

    xlabel_text = ' + '.join([elements_to_keep[j] for j in dimensions_horizontal])
    ylabel_text = ' + '.join([elements_to_keep[j] for j in dimensions_vertical])

    plt.scatter(x_h, x_v, marker='o', s=100, color=y_colors)
    plt.legend(handles=legend_elements, prop={'size': 8})
    #OPTIONAL ANNOTATION
    if annotate:
        for i, txt in enumerate(y):
            x_h = sum([X[i, j] for j in dimensions_horizontal])
            x_v = sum([X[i, j] for j in dimensions_vertical])
            ax.annotate(txt, (x_h, x_v))
            
    fig.text(0.52, 0.03, xlabel_text, ha='center', size=25)
    fig.text(0.07, 0.5, ylabel_text, va='center', rotation='vertical', size=25)
    plt.tick_params(axis='both', labelsize=5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if path_to_save:
        plt.savefig(path_to_save+'low_dimensional.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# def visualise_pca_with_split_labels(X_low_dim, y,
#                                     dimensions = [0, 1],
#                                     method_name = 'pca',
#                                     fixed_color = 'red',
#                                     colormaps = [cm.Greens, cm.Blues, cm.RdPu, cm.Purples, cm.Greys, cm.Oranges],
#                                     annotate = False,
#                                     whether_sort = True,
#                                     figures_path=None):

#     group = y.apply(lambda x: x.split('_')[0])
#     detailed_label = y.apply(lambda x: '_'.join(x.split('_')[1:]))

#     n_groups = group.nunique()
#     colormaps = [fixed_color] + colormaps
#     colormaps = colormaps[:n_groups]

#     color_dicts = []


#     for group_nr, group_name in enumerate(group.unique()):

#         detailed_label_in_group = detailed_label[group == group_name]
#         n_dlig = detailed_label_in_group.nunique()

#         if whether_sort:
#             sorted_labels = sorted(detailed_label_in_group.unique())
#         else:
#             sorted_labels = detailed_label_in_group.unique()

#         if group_nr == 0:
#             color_dict = dict((group_name + '_' + key, fixed_color) for key in sorted_labels)
#         else:
#             cm = colormaps[group_nr]
#             colors = cm(np.linspace(0, 0.99, n_dlig))
#             color_dict = dict((group_name + '_' + key, value) for key, value in zip(sorted_labels, colors))
#         color_dicts.append(color_dict)

#     color_dict = {key: value for d in color_dicts for key, value in d.items()}

#     legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor=color_dict[label],
#                                                   label=label, markersize=15) for label in y]
    
#     x_pca_0 = [X_low_dim[i, dimensions[0]] for i in range(X_low_dim.shape[0])]
#     x_pca_1 = [X_low_dim[i, dimensions[1]] for i in range(X_low_dim.shape[0])]
#     y_colors = np.array([color_dict[el] for el in y])

#     plt.xlabel('PC1')
#     plt.ylabel('PC2')

#     plt.scatter(x_pca_0, x_pca_1, marker='o', s=100, color=y_colors)
#     plt.legend(handles=legend_elements, prop={'size': 8})

#     if figures_path is not None:
#         plt.savefig(figures_path + '_' + method_name + '.png')
    

def visualise_means_pca(X_low_dim, y,
                        figures_name='pca_means',
                        cmap=plt.get_cmap('tab20'),
                        annotate=False, 
                        figures_path=None):

    df = pd.DataFrame(X_low_dim)
    df['Sample_id'] = y.values
    df = df.groupby('Sample_id').mean()
    visualise_pca(df.values, df.index, figures_name=figures_name, cmap=cmap, annotate=annotate, figures_path=figures_path)

def join_datasets_into_one(
                            datasets, 
                            prediction_path_dict,
                            preprocessed_data_path_dict,
                            elements_to_keep,
                            header=0,
                            short_name=True
                            ):
    dfs = []
    labels = []

    for dataset in datasets:
        
        prediction_path = prediction_path_dict[dataset]
        preprocessed_data_path = preprocessed_data_path_dict[dataset]
        
        df = load_prediction(prediction_path, elements_to_keep)
        inds_df = load_target_data(preprocessed_data_path, header=header)
        y_true = create_sample_id_in_target_data(inds_df, column_to_use='name')['Sample_id']

        dfs.append(df)
        labels.append(dataset + '_' + y_true)

    df = pd.concat(dfs)
    y_true = pd.concat(labels)
    if short_name:
        y_true = y_true.apply(lambda x: x.split('_')[0])

    return df, y_true

def visualise_clustering_on_heatmap(X, y, elements_to_keep, colormap=cm.tab20, figsize=(6,5), cbar_pos=None, dendrogram_ratio=0.1, show_classes_names=False, show_legend=False, row_cluster=True, col_cluster=True, figures_path=None):

    y_true = pd.Series(y)
    y_true = y_true.rename('          ')

    heatmap_df = pd.DataFrame(
        data=X,
        columns=elements_to_keep,
        index=np.arange(len(y_true))
    )

    colors = colormap(np.linspace(0, 0.99, y_true.nunique()))
    sorted_labels = sorted(y_true.unique())
    color_dict = dict((key, value) for key, value in zip(sorted_labels, colors))
    if 'corroded' in sorted_labels:
        color_dict['corroded'] = [1., 0., 0., 1.]

    row_colors = y_true.map(color_dict)
    row_colors.index = heatmap_df.index

    cg = sns.clustermap(
        heatmap_df,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_colors=row_colors,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=0.05,
        figsize=figsize,
        norm=LogNorm(),
        cbar_pos=cbar_pos
    )

    #cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    ax = cg.ax_heatmap
    ax.yaxis.set_ticks([])
    ax.tick_params(axis='both', labelsize=25, rotation=90)

    # Row order after clustering
    row_order = cg.dendrogram_row.reordered_ind
    labels = y_true.iloc[row_order].values

    # Find boundaries where label changes
    change_idx = np.where(labels[:-1] != labels[1:])[0] + 1

    # Start + end indices of each block
    block_starts = np.r_[0, change_idx]
    block_ends = np.r_[change_idx, len(labels)]

    # Tick positions = center of each block
    tick_pos = (block_starts + block_ends) / 2
    tick_labels = labels[block_starts]

    ax = cg.ax_row_colors

    if show_legend:
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', markersize=12, linestyle='') for color in color_dict.values()]
        plt.legend(markers, color_dict.keys(), numpoints=1)

    if show_classes_names:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if figures_path is not None:
        plt.savefig(figures_path + 'clustering_heatmap.png', dpi=400)

    plt.show()


def prepare_target_data(
    target_path,
    elements_to_keep,
    how_many_outer_to_remove,
    multiplication_weights,
    preprocessing_method, 
    header=0,
    column_to_use='name',
    normalisation_to_Fe=False,
    return_numpy=False
    ):

    inds_df = load_target_data(target_path, header)

    # ## Preprocessing

    # 0. Keeping track of the records from the same sample.
    # We will keep this info in 'Sample_id' column.

    inds_df = create_sample_id_in_target_data(inds_df, column_to_use)
    
    # 1. Removing 'outer' samples:

    inds_df = remove_outer_samples(inds_df, how_many_outer_to_remove)

    # 2. Removing columns that we don't need.
    # Instead of predicting amounts of all the elements, we will predict only those from elements_to_keep list.

    inds_df = delete_elements(inds_df, elements_to_keep, keep_sample_id=True, keep_name=False)

    # 3. Removing rows with missing values if there are any.

    inds_df = remove_missing_data(inds_df)

    # 4. Setting negative numbers to zeros. (First, checking if there are any.)

    inds_df = set_negative_to_zero(inds_df)

    # 5. Dividing the indicators by weights.

    inds_df = divide_by_weights(inds_df, elements_to_keep, suffix='', weights=multiplication_weights)

    if normalisation_to_Fe:

        # 6. Normalising with respect to Fe.

        inds_df = normalise_to_Fe(inds_df, elements_to_keep, remove_Fe=False, suffixes=[''])

        # 7. Removing Fe.

        elements_to_keep_no_fe = [el for el in elements_to_keep if el != 'Fe']
        inds_df = delete_elements(inds_df, elements_to_keep_no_fe, keep_sample_id=True, keep_name=False)

    # 8. Resetting the index.

    inds_df.reset_index(drop=True, inplace=True)


    # 9. Converting to np.array
    # (Everything except Sample_id column.)

    if normalisation_to_Fe:
        X = np.array(inds_df[elements_to_keep_no_fe].values)
    else:
        X = np.array(inds_df[elements_to_keep].values)

    # 10.  Normalisation / taking logarithm.

    X = transform_data(X, preprocessing_method)


    if return_numpy:
        data_to_return = X


    # 11. Converting to tensors.

    device = get_device()
    X = data_to_device(X, device)

    if return_numpy:
        return X, data_to_return

    else:
        return X


def load_prediction_list(datasets, 
                        prediction_path_dict, 
                        preprocessed_data_path_dict,
                        elements_to_keep,
                        normalisation_to_Fe=False,
                        logarithm=True):
    
    elements_to_keep_no_fe = [el for el in elements_to_keep if el != 'Fe']
    
    df, y_true = join_datasets_into_one(
                                datasets, 
                                prediction_path_dict,
                                preprocessed_data_path_dict,
                                elements_to_keep_no_fe if normalisation_to_Fe else elements_to_keep,
                                short_name = False
                                )
    df.reset_index(inplace=True, drop=True)
    y_true.reset_index(inplace=True, drop=True)
    y_true_short = y_true.apply(lambda x: x.split('_')[0])
    
    dfs = []
    for dataset_name in y_true_short.unique():
        df_dataset = pd.DataFrame(data=df[y_true_short == dataset_name], 
                                  columns=elements_to_keep_no_fe if normalisation_to_Fe else elements_to_keep)
        df_dataset['Sample_id'] = y_true[y_true_short == dataset_name]
        dfs.append(df_dataset)
                
    return dfs

def save_df_list_to_excel(dfs, 
                          excel_prediction_path, 
                          names = ['Konstytucja', 'corroded', 'Merkuriusz', 'Kopernik', 'artificial_inks']
                         ):

    writer = pd.ExcelWriter(excel_prediction_path + 'prediction.xlsx', engine='xlsxwriter')
    
    for i, frame in enumerate(dfs):
       frame.to_excel(writer, sheet_name = names[i], index=False)
        
    writer.close()