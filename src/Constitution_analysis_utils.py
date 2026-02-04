import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
from scipy.spatial.distance import directed_hausdorff
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D


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


def create_class_df(ground_truth_df, book_name):

    if book_name == 'app':
        class_df = ground_truth_df[ground_truth_df['short'].apply(lambda x: x.startswith('APP'))][['Sample_id','OPIS']].copy()
    elif book_name == 'asc':
        class_df = ground_truth_df[ground_truth_df['short'].apply(lambda x: x.startswith('ASC'))][['Sample_id', 'OPIS']].copy()
    elif book_name == 'ml':
        class_df = ground_truth_df[ground_truth_df['short'].apply(lambda x: x.startswith('ML'))][['Sample_id', 'OPIS']].copy()
    elif book_name == 'all':
        class_df = ground_truth_df[['Sample_id', 'OPIS']].copy()

    class_df.columns = ['Code', 'Class name']

    class_df.reset_index(inplace=True, drop=True)

    return class_df


def create_closest_sets(df, class_df, distance_name='Hausdorff distance', distance_function=directed_hausdorff):

    closest_sets = pd.DataFrame(columns = ['Class', 'Closest class', distance_name])

    for chosen_class in class_df['Code'].unique():
        
        chosen_class_distances = pd.DataFrame(columns=['Class', distance_name])
        
        for code in class_df['Code'].unique():
            
            set1 = df[class_df['Code'] == chosen_class].values.copy()
            set2 = df[class_df['Code'] == code].values.copy()
            chosen_class_distances = pd.concat([chosen_class_distances, 
                                                pd.DataFrame([[code, distance_function(set1, set2)[0]]],
                                                            columns=['Class', distance_name])])
            chosen_class_distances.reset_index(inplace=True, drop=True)
            
        which_row = chosen_class_distances.nsmallest(2, distance_name).index[1]
        closest_sets = pd.concat([closest_sets, 
                                 pd.DataFrame([[chosen_class, 
                                                chosen_class_distances.iloc[which_row]['Class'], 
                                                chosen_class_distances.iloc[which_row][distance_name]]],
                                                 columns = ['Class', 'Closest class', distance_name]
                                             )])
    closest_sets = closest_sets.merge(class_df.drop_duplicates(), 
                                      how='left', 
                                      left_on = 'Class', 
                                      right_on='Code')

    closest_sets.drop(columns='Code', inplace=True)
    closest_sets.reset_index(inplace=True, drop=True)

    closest_sets = closest_sets.merge(class_df.drop_duplicates(), 
                                      how='left', 
                                      left_on = 'Closest class', 
                                      right_on='Code')
    closest_sets.rename(columns = {'Class name_x': 'Class name', 'Class name_y': 'Closest class name'},
                        inplace=True)
    closest_sets.drop(columns='Code', inplace=True)

    closest_sets.sort_values(by=distance_name, inplace=True)
    closest_sets.reset_index(inplace=True, drop=True)
    closest_sets.sort_values(by=['Class'], inplace=True)

    return closest_sets


def load_dataset_for_closest_class_assignment(dataset, target_path, Konstytucja_results_path, xrf_path, elements_to_keep, elements_to_keep_xrf):

    if dataset == 'Konstytucja_indicators':
        
        input_data = np.loadtxt(target_path, delimiter=',', skiprows=1, usecols=range(19))
        colnames = pd.read_csv(target_path, nrows=1, header=None)
        df = pd.DataFrame(data=input_data, columns=colnames.iloc[0,:-1])
        df = df[elements_to_keep]
        
    elif dataset == 'Konstytucja_prediction':
        
        input_data = np.loadtxt(Konstytucja_results_path, delimiter=',')
        df = pd.DataFrame(data=input_data, columns=elements_to_keep)
        
    elif dataset == 'XRF':
        
        input_data = np.loadtxt(xrf_path, delimiter=',', skiprows=1, usecols=(2,3,4))
        df = pd.DataFrame(data=input_data, columns=elements_to_keep_xrf)

    else: 
        print('No such dataset!')

    df.reset_index(drop=True, inplace=True)
    return df

def visualise_in_lower_dimension(X_low_dim, 
                                y, 
                                class_df, 
                                dataset,
                                book_name, 
                                marker_shapes = {'AS' : 'v', 'AP': 'd', 'ML': '*'}, 
                                method_name = 'pca',
                                annotate = False, 
                                figures_path=None):

    signatures = [word for word in sorted(list(set(y))) if 'podpis' in word]
    other = sorted(list(set(y).difference(set(signatures))))

    color = cm.gist_ncar(np.linspace(0, 1, len(signatures)))
    grays = np.concatenate([np.array([0.5, 0.5, 0.5, 1]).reshape(1,-1) for _ in range(len(other))], axis=0)
    color = np.concatenate([color, grays])

    legend_elements = []
        
    for label_nr, label in enumerate(signatures + other):
        x_pca_0 = [X_low_dim[i, 0] for i in range(X_low_dim.shape[0]) if y[i] == label]
        x_pca_1 = [X_low_dim[i, 1] for i in range(X_low_dim.shape[0]) if y[i] == label]
        current_codes = [class_df['Code'][i] for i in range(X_low_dim.shape[0]) if y[i] == label]
        if book_name == 'all':
            for nr in range(len(x_pca_0)):
                if nr == 0:
                    legend_elements.append(Line2D([0], [0], color='w', marker='o', markerfacecolor=color[label_nr],
                                                  label=label, markersize=15))
                plt.plot(x_pca_0[nr], x_pca_1[nr], marker=[marker_shapes[cc[:2]] for cc in current_codes][nr], 
                             color=color[label_nr], markersize=15, linestyle='none')
        else:
            plt.plot(x_pca_0, x_pca_1, marker='o', label = label, color=color[label_nr], markersize=15, linestyle='none')
        
    for i in range(X_low_dim.shape[0]):
        if annotate:
            ann = class_df['Code'][i]
            plt.annotate(ann, (X_low_dim[i, 0], X_low_dim[i,1]), size=20)
    if book_name == 'all':
        plt.legend(handles=legend_elements, prop={'size': 24})
        plt.title('ASC: ' + '\u25BC' + ', APP: ' + '\u29EB' + ', ML: ' + '\u2605', fontsize=30)
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 24})

    plt.gcf().set_size_inches((20, 20))
    plt.tight_layout()

    if figures_path is not None:
        plt.savefig(figures_path + dataset + '_' + method_name + '_' + book_name + '_annotate_' + str(annotate) + '.png')