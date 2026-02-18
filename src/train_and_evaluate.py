import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_device
from model import CustomLoss
from model import InksNet
import time
import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
from seaborn import heatmap
import json
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D

with open('../config.json', 'r') as f:
    config = json.load(f)

NORMALISATION_TO_FE = config["normalisation_to_fe"]

if NORMALISATION_TO_FE:
    INPUT_SIZE = len(config["elements_to_keep"])-1 # -1 because we remove Fe at the end of preprocessing
else:
    INPUT_SIZE = len(config["elements_to_keep"]) # no subtraction because we keep Fe

def train_one_epoch(model, loader, loss_fn, optimizer):
    running_loss = 0.

    for i, batch in enumerate(loader):

        inputs, labels = batch ###

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader) # loss per batch
    return avg_loss


def train_model(model, train_loader, val_loader, epochs, loss_fn=CustomLoss(torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1))):

    #loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    epoch_number = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(model, train_loader, loss_fn=loss_fn, optimizer=optimizer)


        running_vloss = 0.0
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                inputs, labels = vdata
                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
                running_vloss += loss.item()
        
        avg_vloss = float(running_vloss / len(val_loader))
        print('LOSS train {} test {}'.format(avg_loss, avg_vloss))
        
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss)

        epoch_number += 1

    return model, train_losses, val_losses


def visualise_losses(train_losses, val_losses):
    plt.plot(train_losses);
    plt.plot(val_losses);
    plt.show()


def evaluate_on_test_set(model, test_loader, loss_fn=CustomLoss(torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1))):
    
    model.eval()
    running_tloss = 0

    for i, tdata in enumerate(test_loader):
        current_inputs, current_labels = tdata
        current_outputs = model(current_inputs)
        loss = loss_fn(current_outputs, current_labels)
        if i == 0:
            outputs = current_outputs
            labels = current_labels
            difference = abs(current_outputs-current_labels)
        else:
            difference = torch.concat((difference, 
                                       abs(current_outputs-current_labels)), 
                                       axis=0)
            labels = torch.concat((labels, current_labels), axis=0)
            outputs = torch.concat((outputs, current_outputs), axis=0)
        running_tloss += loss.item()
        
    mean_loss = running_tloss/len(test_loader)
    return labels, outputs, difference, mean_loss


def save_model(model, models_path, how_many_outer_to_remove, 
                elements_to_keep, preprocessing_method):

    outer_removed = 'outer_removed' if how_many_outer_to_remove>0 else 'outer_not_removed'

    elements = '_'.join(elements_to_keep)
    settings_str = '_' + preprocessing_method + '_' + \
                    outer_removed + '_' + str(how_many_outer_to_remove) + \
                    '_' + elements + '_'

    ts = time.time()
    now = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

    torch.save(model.state_dict(), '/'.join(models_path.split('/')[:-1]) + '/model_regression' + settings_str + now)

def visualise_prediction_against_true(outputs, labels, dimension, lower=-10, upper=10):

    selected_outputs = outputs[:, dimension].cpu().detach().numpy()
    selected_labels = labels[:, dimension].cpu().detach().numpy()
    plt.plot(selected_labels, selected_outputs, '.')
    plt.plot( [lower, upper],[lower, upper], 'r' )
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.show()


def create_means_df(elements_to_keep, y_train, train_order, y_val, val_order):

    train_val_df = pd.DataFrame(y_train)
    train_val_df = pd.concat([train_val_df, 
                                        pd.DataFrame(y_val)])
    train_val_df.columns = elements_to_keep
    train_val_df.insert(0, 'Sample_id', pd.concat([train_order, val_order]))
    train_val_df.reset_index(drop=True, inplace=True)

    mean_df = train_val_df[['Sample_id']].copy()

    for name in elements_to_keep:
        mean_df[name] = train_val_df.groupby('Sample_id')[name].transform('mean')
    mean_df.drop_duplicates('Sample_id', inplace=True)

    mean_df['Sample_id'] = mean_df['Sample_id'].astype(str)
    mean_df.sort_values(by='Sample_id', inplace=True)
    mean_df.reset_index(drop=True, inplace=True)
    return mean_df

def find_and_compare_closest_class(y_test, test_order, elements_to_keep, mean_df):

    y_test_df = pd.DataFrame(y_test)
    y_test_df.columns = elements_to_keep
    y_test_df.insert(0, 'Real sample_id', test_order)
    y_test_df.reset_index(drop=True, inplace=True)

    closest = []
    for i in range(y_test_df.shape[0]):
        which_row = (abs(y_test_df.iloc[i, 1:] - mean_df.iloc[:,1:])).sum(1).idxmin()
        closest.append(mean_df.iloc[which_row]['Sample_id'])

    y_test_df['Closest sample id'] = closest
    return y_test_df, y_test_df[y_test_df['Real sample_id'] == y_test_df['Closest sample id']].shape[0] / y_test_df.shape[0]


def compute_accuracy(outputs_df):
    acc = outputs_df[outputs_df['Real sample_id'] == outputs_df['Closest sample id']].shape[0] / outputs_df.shape[0]
    print('Accuracy: '+str(acc))
    return acc

def compute_precision_and_recall(outputs_df):

    conf_mat = confusion_matrix(outputs_df['Real sample_id'], outputs_df['Closest sample id'], 
                                normalize=None)

    tp_and_fn = conf_mat.sum(1)
    tp_and_fp = conf_mat.sum(0)
    tp = conf_mat.diagonal()

    precision = tp / tp_and_fp
    recall = tp / tp_and_fn

    print('Precision: ' + str(np.nanmean(precision)))
    print('Recall: ' + str(np.nanmean(recall)))

    return np.nanmean(precision), np.nanmean(recall)

def plot_confusion_matrix(true_labels, closest_labels, normalization_in_conf_mat = None, figsize=[15,10], path_to_save = None):

    conf_mat = confusion_matrix(true_labels, closest_labels, 
                                normalize=normalization_in_conf_mat)

    heatmap(conf_mat, cmap=sns.cm.rocket_r, xticklabels=False, yticklabels=False)
    plt.ylabel('True class to which an ink belongs', size=15)
    plt.xlabel("Prediction's closest class", size=15)
    plt.rcParams['figure.figsize'] = figsize
    plt.title('Confusion matrix of classification based on the closest center of class.', 
              size=20, 
              wrap=True)

    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save + 'Confusion_matrix.png')
    plt.show()

def create_min_max_df(y_train, train_order, y_val, val_order, elements_to_keep):

    all_data_df = pd.DataFrame(y_train)
    all_data_df = pd.concat([all_data_df,
                            pd.DataFrame(y_val)])
    all_data_df.columns = elements_to_keep
    all_data_df.insert(0, 'Sample_id', pd.concat([train_order, val_order]))
    all_data_df.reset_index(drop=True, inplace=True)

    min_max_df = all_data_df[['Sample_id']].copy()

    for name in elements_to_keep:
        min_max_df[name + '_max'] = all_data_df.groupby('Sample_id')[name].transform('max')
        
    for name in elements_to_keep:
        min_max_df[name + '_min'] = all_data_df.groupby('Sample_id')[name].transform('min')
        
    min_max_df.drop_duplicates('Sample_id', inplace=True)
    min_max_df.sort_values(by='Sample_id', inplace=True)
    min_max_df.reset_index(drop=True, inplace=True)

    max_df = min_max_df.iloc[:,1:(1+len(elements_to_keep))]
    max_df['Sample_id'] = min_max_df['Sample_id']
    min_df = min_max_df.iloc[:,1+len(elements_to_keep):]
    min_df['Sample_id'] = min_max_df['Sample_id']
    return min_df, max_df, min_max_df

def create_mean_plus_sd_df(y_train, train_order, y_val, val_order, elements_to_keep, how_many_sd=2):

    all_data_df = pd.DataFrame(y_train)
    all_data_df = pd.concat([all_data_df, 
                            pd.DataFrame(y_val)])
    all_data_df.columns = elements_to_keep
    all_data_df.insert(0, 'Sample_id', pd.concat([train_order, val_order]))
    all_data_df.reset_index(drop=True, inplace=True)
    mean_sd_df = all_data_df[['Sample_id']].copy()

    for name in elements_to_keep:
        mean_sd_df[name + '_mean'] = all_data_df.groupby('Sample_id')[name].transform('mean')
        
    for name in elements_to_keep:
        mean_sd_df[name + '_sd'] = all_data_df.groupby('Sample_id')[name].transform('std')

    mean_sd_df.drop_duplicates('Sample_id', inplace=True)   
    mean_sd_df.sort_values(by='Sample_id', inplace=True)
    mean_sd_df.reset_index(drop=True, inplace=True)

    upper_bound_df = mean_sd_df.iloc[:,1:(1+len(elements_to_keep))].values + \
                    how_many_sd*mean_sd_df.iloc[:,1+len(elements_to_keep):].values
    upper_bound_df = pd.DataFrame(upper_bound_df, columns=elements_to_keep)
    upper_bound_df['Sample_id'] = mean_sd_df[['Sample_id']]

    lower_bound_df = mean_sd_df.iloc[:,1:(1+len(elements_to_keep))].values - \
                    how_many_sd*mean_sd_df.iloc[:,1+len(elements_to_keep):].values
    lower_bound_df = pd.DataFrame(lower_bound_df, columns=elements_to_keep)
    lower_bound_df['Sample_id'] = mean_sd_df[['Sample_id']]

    return lower_bound_df, upper_bound_df


def is_inside_interval(outputs, test_order, lower_bound_df, upper_bound_df):

    sd_res_list = []

    for test_example, sample_id in zip(outputs, test_order):
        
        greater = test_example.cpu().detach().numpy() > \
        np.array(lower_bound_df.loc[lower_bound_df['Sample_id'] == sample_id, lower_bound_df.columns != 'Sample_id'])
        
        less = test_example.cpu().detach().numpy() < \
        np.array(upper_bound_df.loc[upper_bound_df['Sample_id'] == sample_id, upper_bound_df.columns != 'Sample_id'])
        
        res = np.logical_and(greater, less)
        
        sd_res_list.append(res)

    return np.array(sd_res_list), np.array(sd_res_list).mean()

def visualise_min_max_intervals(outputs, test_order, elements_to_keep, min_max_df, min_max_res_list, element_nr, path_to_save=None):

    outputs_df = pd.DataFrame(outputs.cpu().detach().numpy())
    outputs_df.columns = elements_to_keep
    outputs_df.insert(0, 'Real sample_id', test_order)
    outputs_df['Real sample_id'] = outputs_df['Real sample_id'].astype(str)
    outputs_df.reset_index(drop=True, inplace=True)

    percentage_correct = np.round(100*(np.array(min_max_res_list).mean(0))[0, element_nr], 1)

    sorted_min_max_df = min_max_df.sort_values(min_max_df.columns[element_nr+1])
    sorted_min_max_df.reset_index(drop=True, inplace=True)

    #plt.rcParams['figure.figsize'] = [8, 8]

    for sample_id in range(sorted_min_max_df.shape[0]):
        plt.vlines(x=sample_id, ymin=sorted_min_max_df.iloc[sample_id, element_nr+len(elements_to_keep)+1], 
                      ymax=sorted_min_max_df.iloc[sample_id, element_nr+1], 
                      color="blue", linewidth=1)

    for row in range(outputs_df.shape[0]):
        new_index = sorted_min_max_df[sorted_min_max_df['Sample_id'] == outputs_df['Real sample_id'][row]].index
        plt.plot(new_index, outputs_df.iloc[row, element_nr+1], 'ro', markersize=3)
            
    plt.xticks([], [])
    plt.xlabel('Groups IDs',size=15)
    plt.ylabel('Value',size=15)
    plt.title(str(outputs_df.columns[element_nr+1]) + ': '+str(percentage_correct) +'% inside min-max interval', size=25)
    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save + '_min_max_intervals_' + str(outputs_df.columns[element_nr+1]))
    plt.show()

def visualise_mean_plus_minus_sd_intervals(outputs, 
                                            test_order, 
                                            elements_to_keep, 
                                            lower_bound_df, 
                                            upper_bound_df, 
                                            how_many_sd,
                                            sd_res_list, 
                                            element_nr, 
                                            path_to_save=None):

    outputs_df = pd.DataFrame(outputs.cpu().detach().numpy())
    outputs_df.columns = elements_to_keep
    outputs_df.insert(0, 'Real sample_id', test_order)
    outputs_df['Real sample_id'] = outputs_df['Real sample_id'].astype(str)
    outputs_df.reset_index(drop=True, inplace=True)

    percentage_correct = np.round(100*(np.array(sd_res_list).mean(0))[0, element_nr], 1)

    sorted_upper_bound_df = upper_bound_df.sort_values(upper_bound_df.columns[element_nr])
    sorted_upper_bound_df.reset_index(drop=True, inplace=True)
    sorted_lower_bound_df = lower_bound_df.iloc[upper_bound_df.sort_values(upper_bound_df.columns[element_nr]).index,:]
    sorted_lower_bound_df.reset_index(drop=True, inplace=True)

    #plt.rcParams['figure.figsize'] = [8, 8]

    for sample_id in range(sorted_lower_bound_df.shape[0]):
        plt.vlines(x=sample_id, ymin=sorted_lower_bound_df.iloc[sample_id, element_nr], 
                      ymax=sorted_upper_bound_df.iloc[sample_id, element_nr], 
                      color="blue", linewidth=1)
        
    for row in range(outputs_df.shape[0]):
        new_index = sorted_upper_bound_df[sorted_upper_bound_df['Sample_id'] == outputs_df['Real sample_id'][row]].index
        plt.plot(new_index, outputs_df.iloc[row, element_nr+1], 'ro', markersize=3)
            
    plt.xticks([], [])
    plt.xlabel('Groups IDs',size=15)
    plt.ylabel('Value',size=15)
    plt.title(str(outputs_df.columns[element_nr+1]) + ': '+str(percentage_correct) +'% inside +-' + str(how_many_sd) + ' sd confidence interval', size=25)
    
    if path_to_save is not None:
        plt.savefig(path_to_save + '_mean_plus_minus_sd_intervals_' + str(outputs_df.columns[element_nr+1]))
    plt.show()