import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
from scipy.spatial.distance import directed_hausdorff

class InksDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index, :], self.y[index, :]
    
    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0]
        return self.X.shape[0]  

def load_training_data(data_path, indicators_suffix='_i', inks_suffix='_a'):

    inDKs_df = pd.read_csv(data_path)
    inks_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith(inks_suffix)]]
    inks_df.columns = [col.split('_')[0] for col in inks_df.columns]
    inds_df = inDKs_df[[col for col in inDKs_df.columns if col.endswith(indicators_suffix)]]
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

def delete_elements_v2(inDKs_df, elements_to_keep):
    inDKs_df = inDKs_df[elements_to_keep + ['Sample_id']]
    return inDKs_df

def delete_elements_v3(inDKs_df, elements_to_keep):
    inDKs_df = inDKs_df[elements_to_keep]
    return inDKs_df

def remove_missing_data(inDKs_df):
    how_many_nans = inDKs_df.shape[0] - inDKs_df.dropna().shape[0]
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

def multiply_by_weights(inDKs_df, elements_to_keep, suffix='',
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):
    inDKs_df = inDKs_df.copy()
    weights = [el/sum(weights) for el in weights]
    columns_to_transform = [el + suffix for el in elements_to_keep]
    for i, col in enumerate(columns_to_transform):
        inDKs_df[col] = weights[i]*inDKs_df[col]
    return inDKs_df

def divide_by_weights(inDKs_df, elements_to_keep, suffix='',
                        weights=[1, 1, 1, 10, 19, 20, 17, 9, 20, 1]):
    inDKs_df = inDKs_df.copy()
    weights = [el/sum(weights) for el in weights]
    columns_to_transform = [el + suffix for el in elements_to_keep]
    for i, col in enumerate(columns_to_transform):
        inDKs_df[col] = inDKs_df[col]/weights[i]
    return inDKs_df

def normalize_to_Fe(inDKs_df, elements_to_keep, remove_Fe=False):
    inDKs_df = inDKs_df.copy()
    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]
    columns_to_keep_inks = [el + '_a' for el in elements_to_keep]
    for i, col in enumerate(columns_to_keep_inds):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe_i']
    for i, col in enumerate(columns_to_keep_inks):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe_a']
    if remove_Fe:
        inDKs_df = inDKs_df.drop(columns=['Fe_a', 'Fe_i'])
    return inDKs_df

def normalize_to_Fe_v2(inDKs_df, elements_to_keep, remove_Fe=False):
    inDKs_df = inDKs_df.copy()
    for i, col in enumerate(elements_to_keep):
        inDKs_df[col] = inDKs_df[col] / inDKs_df['Fe']
    if remove_Fe:
        inDKs_df = inDKs_df.drop(columns=['Fe'])
    return inDKs_df

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

def transform_data(X, preprocessing_method):
    
    def adjusted_log_transform(input_array):
        res = np.where(input_array>0, np.log(input_array), 0.)
        return res
    
    if preprocessing_method == 'normalization':

        X = (X - np.min(X, axis=0))/np.std(X, axis=0)
    
    elif preprocessing_method == 'logarithm':
        
        X = adjusted_log_transform(X)

    elif preprocessing_method == 'logarithm_and_normalization':
        
        #logarithm
        X = adjusted_log_transform(X)
        
        #normalization
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

    #continuous cmap
    colors = [cmap(i / n_colors) for i in range(n_colors)]

    #discrete cmap with 20 colors
    #colors = [cmap(i%20) for i in range(n_colors)] 

    np.random.seed(seed)
    color_dict = dict(zip(np.random.permutation(X_sample_ids.unique()), colors))

    fig, ax = plt.subplots(figsize=(7,7))
    for i, name in enumerate(X_sample_ids):
        ax.plot(X_pca[i, 0], X_pca[i, 1], marker='o', linestyle='', ms=8, color=color_dict[name])
        
    for j, name in enumerate(means_sample_ids):
        ax.plot(means_pca[j, 0], means_pca[j, 1], marker='X', linestyle='', ms=20, color=color_dict[name]) #marker's colorful filling
        ax.plot(means_pca[j, 0], means_pca[j, 1], marker='X', linestyle='', ms=20, color='black', fillstyle='none') #marker's black frame

    ax.legend()

    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    if x_lower is not None and x_upper is not None:
        ax.set_xlim(x_lower, x_upper)
    if y_lower is not None and y_upper is not None:
        ax.set_ylim(y_lower, y_upper)

    plt.tight_layout()
        
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300)

def visualise_train_val_test_distributions(train_data, 
                                            val_data, 
                                            test_data,
                                            elements_to_keep,
                                            title='',
                                            lower_x_lim=-5,
                                            upper_x_lim=15,
                                            lower_y_lim=0,
                                            upper_y_lim=100,
                                            path_to_save=None):

    fig, axes = plt.subplots(len(elements_to_keep)//2, 2, figsize=(7, 7), sharey=True, sharex=True) 

    axes = axes.flatten()

    for i in range(len(elements_to_keep)):

        axes[i].hist(train_data[:,i], bins=80, label='training', color='#007C91')
        axes[i].hist(val_data[:,i], bins=80, label='validation', color='#E66100')
        axes[i].hist(test_data[:,i], bins=80, label='test', color='#4B4B4B')
        axes[i].set_title(title + elements_to_keep[i], x=0.5, y=0.75)
        axes[i].set_xlim([lower_x_lim, upper_x_lim])
        axes[i].set_ylim([lower_y_lim, upper_y_lim])

    fig.supxlabel('Logarithm of relative quantity')
    plt.legend(bbox_to_anchor=(1.6,2.5))

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', dpi=300)




def load_target_data(target_path, elements_to_keep, header=0):
    inds_df = pd.read_csv(target_path, usecols = elements_to_keep, header=header)
    if 'name' in inds_df.columns:
        inds_df['Sample_id'] = inds_df['name'].apply(lambda x: x.split('_')[0] if len(x.split('_'))==2 else x.split('_')[0] + x.split('_')[1])
        inds_df.drop(columns=['name'], inplace=True)
    inds_df.reset_index(drop=True, inplace=True)
    return inds_df

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
                    annotate = False, 
                    whether_sort = True,
                    figures_path=None):

    colors = cm.tab20(np.linspace(0, 0.99, y.nunique()))

    if whether_sort:
        sorted_labels = sorted(y.unique())
    else:
        sorted_labels = y.unique()

    color_dict = dict((key, value) for key, value in zip(sorted_labels, colors))
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
        plt.savefig(figures_path + figures_name + '_' + 'PC' + str(dimensions[0]+1) + '_' + 'PC' + str(dimensions[1]+1) + '.png')

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
    

def visualise_means_pca(X, y, elements_to_keep,
                        method_name = 'pca_means',
                        annotate = False, 
                        figures_path=None):

    df = pd.DataFrame(X, columns=elements_to_keep)
    df['Sample_id'] = y
    df = df.groupby('Sample_id').mean()
    visualise_pca(df.values, df.index, method_name=method_name, annotate=annotate, figures_path=figures_path)

def visulise_clustering_on_heatmap(X, y, elements_to_keep, figures_path=None, show_classes_names=False):

    y_true = pd.Series(y)
    y_true = y_true.rename('ID')

    heatmap_df = pd.DataFrame(
        data=X,
        columns=elements_to_keep,
        index=np.arange(len(y_true))
    )

    colors = cm.tab20(np.linspace(0, 0.99, y_true.nunique()))
    sorted_labels = sorted(y_true.unique())
    color_dict = dict((key, value) for key, value in zip(sorted_labels, colors))

    row_colors = y_true.map(color_dict)
    row_colors.index = heatmap_df.index

    cg = sns.clustermap(
        heatmap_df,
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        dendrogram_ratio=0.1,
        colors_ratio=0.05,
        figsize=(7, 7),
        cbar_pos=None
    )

    #cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    ax = cg.ax_heatmap
    ax.yaxis.set_ticks([])

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

    if show_classes_names:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    plt.savefig(figures_path + 'clustering_heatmap.png', dpi=400)