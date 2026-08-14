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
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_recall_fscore_support
import seaborn as sns
import json
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from scipy.stats import probplot
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from collections import Counter
from astropy.convolution import Gaussian2DKernel, convolve
import joblib
from pathlib import Path

import torch.optim as optim
import optuna

with open('../config.json', 'r') as f:
    config = json.load(f)

NORMALISATION_TO_FE = config["normalisation_to_Fe"]

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

    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.000917009720506624, weight_decay=3.0268226950713e-08) #optuna_l1_71_3000
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.184633313770491e-05, weight_decay=6.411548994103367e-05)


    model, train_losses, val_losses = train_model_optuna(model=model,
                                                        train_loader=train_loader,
                                                        val_loader=val_loader,
                                                        epochs=epochs,
                                                        optimizer=optimizer, 
                                                        loss_fn=loss_fn)

    return model, train_losses, val_losses


def train_model_optuna(model, train_loader, val_loader, epochs, optimizer, loss_fn=CustomLoss(torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1))):

    epoch_number = 0

    train_losses = []
    corrected_train_losses = []
    val_losses = []

    for epoch in range(epochs):

        if epoch % 50 == 0:
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

        ########
        corrected_running_loss = 0.0
        with torch.no_grad():
            for i, tdata in enumerate(train_loader):
                inputs, labels = tdata
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                corrected_running_loss += loss.item()
        corrected_avg_running_loss = float(corrected_running_loss / len(train_loader))
        #########

        if epoch % 50 == 0:
            #print('LOSS train {} test {}'.format(avg_loss, avg_vloss))
            print('LOSS train {} test {}'.format(corrected_avg_running_loss, avg_vloss))
        
        train_losses.append(avg_loss)
        corrected_train_losses.append(corrected_avg_running_loss)
        val_losses.append(avg_vloss)

        epoch_number += 1

    #return model, train_losses, val_losses
    return model, corrected_train_losses, val_losses


def visualise_losses(train_losses, val_losses, figsize=(12,5), path_to_save=None):

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train_losses, label='training', alpha=0.5)
    ax.plot(val_losses, label='validation', alpha=0.5)

    fig.text(0.52, 0.03, "Epoch", ha='center', size=25)
    fig.text(0.07, 0.5, "L1 error of prediction", va='center', rotation='vertical', size=25)
    plt.tick_params(axis='both', labelsize=5)
    plt.legend()

    if path_to_save:
        plt.savefig(path_to_save+'losses.png', dpi=300)
    plt.show()
    plt.close(fig)


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


def load_or_train_model(model_path, model, X_train, y_train):
    model_path = Path(model_path)

    if model_path.exists():
        print(f"Loading model from {model_path}")
        return joblib.load(model_path)

    print("Model not found. Training...")
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")
    return model


# ARCHITECTURE OPTIMISATION

# -----------------------------
# Model builder
# -----------------------------
def build_model(trial, input_size=INPUT_SIZE):

    n_layers = trial.suggest_int("n_layers", 2, 4)

    activation_name = trial.suggest_categorical(
        "activation", ["relu", "tanh", "gelu"]
    )

    dropout = trial.suggest_float("dropout", 0.04, 0.12)

    layers = []
    in_features = input_size

    for i in range(n_layers):

        out_features = trial.suggest_int(f"n_units_l{i}", 50, 900, log=True)

        layers.append(nn.Linear(in_features, out_features))

        if activation_name == "relu":
            layers.append(nn.ReLU())
        elif activation_name == "tanh":
            layers.append(nn.Tanh())
        else:
            layers.append(nn.GELU())
        #layers.append(nn.GELU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        in_features = out_features

    layers.append(nn.Linear(in_features, input_size))

    return nn.Sequential(*layers)


# VISUALISATIONS

# BASIC DIAGNOSTIC PLOTS

def visualise_prediction_against_true(outputs, labels, dimension, xlabel='True value', ylabel='Predicted value', title=''):

    selected_outputs = outputs[:, dimension].cpu().detach().numpy()
    selected_labels = labels[:, dimension].cpu().detach().numpy()

    lower = selected_outputs.min()
    upper = selected_outputs.max()

    plt.plot(selected_labels, selected_outputs, '.')
    plt.plot( [lower, upper],[lower, upper], 'r' )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def compute_metrics(outputs, labels):

    assert outputs.shape == labels.shape
    n_dims = outputs.shape[1]

    mae, rmse, r2, bias, max_error = [], [], [], [], []

    for i in range(n_dims):
        y_true = labels[:, i]
        y_pred = outputs[:, i]

        mae.append(mean_absolute_error(y_true, y_pred))
        rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2.append(r2_score(y_true, y_pred))
        bias.append(np.mean(y_pred - y_true))
        max_error.append(np.max(np.abs(y_pred - y_true)))

    l2_error = np.linalg.norm(outputs - labels, axis=1)

    summary = {
        "mae": np.array(mae),
        "rmse": np.array(rmse),
        "r2": np.array(r2),
        "bias": np.array(bias),
        "mean_mae": float(np.mean(mae)),
        "mean_rmse": float(np.mean(rmse)),
        "mean_r2": float(np.mean(r2)),
        "mean_l2": float(np.mean(l2_error)),
        "max_error": np.array(max_error)
    }

    return summary

def plot_pred_vs_gt(outputs, labels, elements_to_keep, dims_to_keep='all', nrows=2, figsize=(12, 5), xlabel='True value', ylabel='Prediction', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    fig, axes = plt.subplots(nrows, n_dims//nrows + 1 if nrows==2 else n_dims//nrows, figsize=figsize)
    if len(dims_to_keep)>1:
        axes = axes.flatten()

    metrics = compute_metrics(outputs, labels)

    for i in range(n_dims):

        if len(dims_to_keep)>1:
            ax = axes[i]
        else:
            ax = axes

        y_true = labels[:, i]
        y_pred = outputs[:, i]

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'black')

        ax.scatter(y_true, y_pred, alpha=0.5)

        title = elements_to_keep[i]

        text = f"R$^2$={metrics['r2'][i]:.2f} \nMAE={metrics['mae'][i]:.2f} \nRMSE={metrics['rmse'][i]:.2f}"

        ax.text(min_val, max_val-(0.10+0.16*(nrows-1))*(max_val-min_val), text)
        #ax.text(min_val, y_max_max-0.45*nrows, text)

        ax.set_title(title, size=10)
        ax.tick_params(axis='both', labelsize=5)

    if nrows==2:
        axes[-1].set_axis_off()

    fig.text(0.52, 0.0, xlabel, ha='center', size=25)
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', size=25)

    #plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save+'prediction_vs_true.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_residuals(outputs, labels, elements_to_keep, dims_to_keep='all', nrows=2, figsize=(12,5), xlabel='True value', ylabel='Residual', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    fig, axes = plt.subplots(nrows, n_dims//nrows + 1 if nrows==2 else n_dims//nrows, figsize=figsize, sharey=True)
    if len(dims_to_keep)>1:
        axes = axes.flatten()

    y_max_max = (outputs-labels).max().max()

    for i in range(n_dims):

        if len(dims_to_keep)>1:
            ax = axes[i]
        else:
            ax = axes

        y_true = labels[:, i]
        y_pred = outputs[:, i]

        min_val = y_true.min()
        max_val = y_true.max()
        mean_res = np.mean(y_pred-y_true)

        ax.plot([min_val, max_val], [mean_res, mean_res], 'black')

        ax.scatter(y_true, y_pred-y_true, alpha=0.25)

        title = elements_to_keep[i]

        y_max = (y_pred-y_true).max()
        y_min = (y_pred-y_true).min()
        text = 'Mean residual: \n' + str(np.round(mean_res, 3))
        ax.text(min_val, y_max_max-0.25-0.65*(nrows-1), text)

        ax.tick_params(axis='both', labelsize=5)
        ax.set_title(title, size=10)

    if nrows==2:
        axes[-1].set_axis_off()

    fig.text(0.5, 0.0, xlabel, ha='center', size=25)
    fig.text(0.06, 0.5, ylabel, va='center', rotation='vertical', size=25)

    #plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save+'residuals.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_error_distributions(outputs, labels, elements_to_keep, dims_to_keep='all', nrows=2, figsize=(10, 5), xlabel='Residual', ylabel='Count', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    fig, axes = plt.subplots(nrows, n_dims//nrows + 1 if nrows==2 else n_dims//nrows, figsize=figsize, sharey=True, sharex=True)
    if len(dims_to_keep)>1:
        axes = axes.flatten()

    y_max_max = (outputs-labels).max().max()

    for i in range(n_dims):

        if len(dims_to_keep)>1:
            ax = axes[i]
        else:
            ax = axes
            
        y_true = labels[:, i]
        y_pred = outputs[:, i]


        residuals = y_pred-y_true
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set(ylabel='')

        title = elements_to_keep[i]

        ax.set_title(title, size=10)
        ax.tick_params(axis='both', labelsize=5)


    if nrows==2:
        axes[-1].set_axis_off()

    fig.text(0.52, 0.0, xlabel, ha='center', size=25)
    fig.text(0.06, 0.5, ylabel, va='center', rotation='vertical', size=25)

    #plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save+'error_distributions.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_qq(outputs, labels, elements_to_keep, dims_to_keep='all', nrows=2, figsize=(16, 5), xlabel='Theoretical quantiles', ylabel='Prediction \n quantiles', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    fig, axes = plt.subplots(nrows, n_dims//nrows + 1 if nrows==2 else n_dims//nrows, figsize=figsize, sharey=True, sharex=True)
    if len(dims_to_keep)>1:
        axes = axes.flatten()

    for i in range(n_dims):

        if len(dims_to_keep)>1:
            ax = axes[i]
        else:
            ax = axes
            
        y_true = labels[:, i]
        y_pred = outputs[:, i]

        norm_residuals = ((y_pred-y_true)-np.mean(y_pred-y_true))/np.std(y_pred-y_true)

        ret1, ret2 = probplot(norm_residuals, dist="norm")
        osm, osr = ret1
        ax.scatter(osm, osr, alpha=0.5)
        ax.set(ylabel='')

        title = elements_to_keep[i]

        ax.set_title(title, size=10)
        ax.tick_params(axis='both', labelsize=5)

    if nrows==2:
        axes[-1].set_axis_off()

    fig.text(0.52, 0.0, xlabel, ha='center', size=25)
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', size=25)

    #plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save+'qq_plot.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_error_boxplot(outputs, labels, elements_to_keep, dims_to_keep='all', figsize=(12, 5), xlabel='', ylabel='Residual', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    residuals_all = outputs - labels

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=residuals_all, ax=ax)
    fig.text(0.52, 0.03, xlabel, ha='center', size=25)
    fig.text(0.07, 0.5, ylabel, va='center', rotation='vertical', size=25)
    ax.set_xticklabels(elements_to_keep, size=25)
    ax.tick_params(axis='y', labelsize=5)
    plt.ylim(-6, 6)

    if path_to_save:
        plt.savefig(path_to_save+'error_boxplot.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_error_violinplot(outputs, labels, elements_to_keep, dims_to_keep='all', figsize=(12, 5), xlabel='', ylabel='Residual', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    residuals_all = outputs - labels

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=residuals_all, ax=ax)
    fig.text(0.52, 0.03, xlabel, ha='center', size=25)
    fig.text(0.07, 0.5, ylabel, va='center', rotation='vertical', size=25)
    ax.set_xticklabels(elements_to_keep, size=25)
    ax.tick_params(axis='y', labelsize=5)
    plt.ylim(-6, 6)

    if path_to_save:
        plt.savefig(path_to_save+'error_violinplot.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_correlation_heatmaps(outputs, labels, elements_to_keep, figsize=(12, 5), xlabel='', ylabel='Residual', cluster=False, path_to_save=None):

    def clustered_corr(data, method='average'):
        corr = np.corrcoef(data, rowvar=False)
        Z = linkage(corr, method=method)
        order = leaves_list(Z)
        return corr[order][:, order]

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    if cluster:
        ax = sns.heatmap(clustered_corr(labels), annot=False, cmap='rocket_r', vmin=-0.6, vmax=1)
    else:
        ax = sns.heatmap(np.corrcoef(labels, rowvar=False), annot=False, cmap='rocket_r', vmin=-0.6, vmax=1)

    ax.figure.axes[-1].yaxis.label.set_size(10)
    plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = True, labeltop=True)

    ax.set_xticklabels(elements_to_keep, size=10)
    ax.set_yticklabels(elements_to_keep, size=10)

    plt.subplot(1, 2, 2)
    if cluster:
        ax = sns.heatmap(clustered_corr(outputs), annot=False, cmap='rocket_r', vmin=-0.6, vmax=1)
    else:
        ax = sns.heatmap(np.corrcoef(outputs, rowvar=False), annot=False, cmap='rocket_r', vmin=-0.6, vmax=1)

    ax.figure.axes[-1].yaxis.label.set_size(10)
    ax.set_xticklabels(elements_to_keep, size=10)
    ax.set_yticklabels(elements_to_keep, size=10)


    plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = True, labeltop=True)

    fig.text(0.47, 0.05, "     True correlations               Prediction correlations", ha='center', size=25)

    if path_to_save:
        plt.savefig(path_to_save+'correlation_heatmap.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_correlation_heatmaps_for_target_data(prediction, elements_to_keep, dataset_name='', path_to_save=None):

    def clustered_corr(data, method='average'):
        corr = np.corrcoef(data, rowvar=False)
        Z = linkage(corr, method=method)
        order = leaves_list(Z)
        return corr[order][:, order], order

    corr_mat, order = clustered_corr(prediction)
    order = [elements_to_keep[i] for i in order]
    corr_mat = pd.DataFrame(corr_mat, columns=order, index=order)
    sns.heatmap(corr_mat, annot=False, cmap='rocket_r', vmin=-1, vmax=1)

    plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = True, labeltop=True)
    if path_to_save:
        plt.savefig(path_to_save + dataset_name + '_correlation_heatmap.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_l1_error(outputs, labels, figsize=(12, 5), path_to_save=None):

    l2_error = np.sum(np.abs(outputs - labels), axis=1)

    true_norm = np.sum(np.abs(labels), axis=1)

    fig = plt.figure(figsize=figsize)
    plt.scatter(true_norm, l2_error, alpha=0.5)
    plt.xlabel("Sum of true magnitudes", size=25)
    plt.ylabel("L1 error of prediction", size=25)
    plt.tick_params(axis='both', labelsize=5)

    if path_to_save:
        plt.savefig(path_to_save+'global_error_vs_magnitude.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_l1_error_with_density(outputs, labels, figsize=(12, 5), path_to_save=None):

    l2_error = np.sum(np.abs(outputs - labels), axis=1)

    true_norm = np.sum(np.abs(labels), axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    #ax.plot(true_norm[~inside], l2_error[~inside], 'bo', mfc='none')
    ax.plot(true_norm, l2_error, 'bo', mfc='none')

    H, xedges, yedges = np.histogram2d(true_norm, l2_error, bins=(50,40))
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])

    kernel = Gaussian2DKernel(1.)
    H = convolve(H, kernel)

    levels = np.linspace(H.min(), H.max(), 10)
    levels = levels[1:]  # skip the outermost (lowest) level

    clevels = ax.contour(xmesh, ymesh, H.T, levels=levels, lw=.9, cmap='jet')

    p = clevels.get_paths()
    inside = np.full_like(true_norm, False, dtype=bool)
    for level in p:
    #     inside |= level.contains_points(zip(*(true_norm, l2_error)))
        points = np.column_stack((true_norm, l2_error))
        inside |= level.contains_points(points)

    fig.text(0.52, 0.03, "Sum of true magnitudes", ha='center', size=25)
    fig.text(0.07, 0.5, "L1 error of prediction", va='center', rotation='vertical', size=25)
    plt.tick_params(axis='both', labelsize=5)

    if path_to_save:
        plt.savefig(path_to_save+'global_error_vs_magnitude_with_density.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_pca_projection(outputs, labels, figsize=(8, 5), path_to_save=None, max_arrows=200):

    # Fit PCA on ground truth only (important for fair comparison)
    pca = PCA(n_components=2)
    pca.fit(labels)

    labels_2d = pca.transform(labels)
    outputs_2d = pca.transform(outputs)

    fig = plt.figure(figsize=figsize)

    plt.scatter(labels_2d[:, 0], labels_2d[:, 1], alpha=0.5, label="Ground Truth")
    plt.scatter(outputs_2d[:, 0], outputs_2d[:, 1], alpha=0.5, label="Prediction")

    from sklearn.metrics import pairwise_distances

    d_true = pairwise_distances(labels_2d)
    d_pred = pairwise_distances(outputs_2d)

    # Normalize (avoid scale issues)
    d_true /= np.mean(d_true)
    d_pred /= np.mean(d_pred)

    alignment_score = np.mean(np.abs(d_true - d_pred))

    print(f"PCA alignment score (lower is better): {alignment_score:.4f}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (2D)")
    plt.legend()

    if path_to_save:
        plt.savefig(path_to_save+'pca_projection_test_set.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# CLASSIFICATION-BASED STATISTICS AND VISUALISATIONS PART 1

def plot_topk_confusion(y_true, y_pred, top_k=20, figsize=(7, 5), path_to_save=None):

    # get most frequent classes
    counts = Counter(y_true)
    top_classes = [c for c, _ in counts.most_common(top_k)]
    print(top_classes)

    mask = np.isin(y_true, top_classes)
    y_true_k = y_true[mask]
    y_pred_k = y_pred[mask]

    plt.figure(figsize=figsize)

    cm = confusion_matrix(y_true_k, y_pred_k, labels=top_classes)
    ax=sns.heatmap(cm, cmap=sns.cm.rocket_r, xticklabels=False, yticklabels=False, cbar=True)

    #plt.imshow(cm, cmap=sns.cm.rocket_r)

    #plt.title(f"Top-{top_k} confusion matrix", size=25)
    plt.xlabel("Prediction's closest class", size=25)
    plt.ylabel("True class", size=25)
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()

    if path_to_save:
        plt.savefig(path_to_save+'top_k_confusion_matrix.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_prf_distribution_subplots(y_true, y_pred, figsize=(16, 5), xlabel='', ylabel='Count', path_to_save=None):

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)

    axes[0].hist(precision, bins=30)
    axes[0].set_xlabel("Precision", size=25)

    axes[1].hist(recall, bins=30)
    axes[1].set_xlabel("Recall", size=25)

    axes[2].hist(f1, bins=30)
    axes[2].set_xlabel("F1", size=25)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=5)

    fig.text(0.5, 0.9, xlabel, ha='center', size=25)
    fig.text(0.09, 0.5, ylabel, va='center', rotation='vertical', size=25)

    if path_to_save:
        plt.savefig(path_to_save+'prec_rec_f1_distributions.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_freq_vs_f1(y_true, y_pred, figsize=(15, 5), xlabel='Class frequency', path_to_save=None):

    counts = Counter(y_true)
    labels = np.unique(y_true)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )

    freqs = np.array([counts[l] for l in labels])

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)

    #axes[0].hist2d(freqs, precision, bins=30)
    axes[0].scatter(freqs, precision, alpha = 0.2, s=[0.7*len(freqs)])
    axes[0].set_ylabel("Precision", size=25)

    #axes[1].hist2d(freqs, recall, bins=30)
    axes[1].scatter(freqs, recall, alpha = 0.2, s=[0.7*len(freqs)])
    axes[1].set_ylabel("Recall", size=25)

    #axes[2].hist2d(freqs, f1, bins=30)
    axes[2].scatter(freqs, f1, alpha = 0.2, s=[0.7*len(freqs)])
    axes[2].set_ylabel("F1", size=25)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=5)

    fig.text(0.5, 0.0, xlabel, ha='center', size=25)
    #fig.colorbar(label="Number of classes")
    #fig.text(0.05, 0.5, ylabel, va='center', rotation='vertical', size=25)

    # plt.figure()
    # plt.scatter(freqs, f1, alpha = 0.2)
    # plt.ylabel("F1 score")
    # plt.title("Class frequency vs F1")
    if path_to_save:
        plt.savefig(path_to_save+'freq_vs_f1.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# CLASSIFICATION-BASED STATISTICS AND VISUALISATIONS PART 2

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

def get_sample_number_in_group(original_labels):
    return np.array(original_labels.apply(lambda x: int(x.split('_')[-1])).values)

def plot_confusion_matrix(true_labels, closest_labels, normalization_in_conf_mat = None, figsize=(7, 5), path_to_save = None):

    conf_mat = confusion_matrix(true_labels, closest_labels, 
                                normalize=normalization_in_conf_mat)

    plt.figure(figsize=figsize)
    ax=sns.heatmap(conf_mat, cmap=sns.cm.rocket_r, xticklabels=False, yticklabels=False, cbar=True)
    ax.figure.axes[-1].yaxis.label.set_size(10)
    plt.ylabel('True class', size=25)
    plt.xlabel("Prediction's closest class", size=25)
    #plt.rcParams['figure.figsize'] = figsize
    plt.title('', 
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
    plt.xlabel('Groups IDs',size=25)
    plt.ylabel('Value',size=25)
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
    plt.xlabel('Groups IDs',size=25)
    plt.ylabel('Value',size=25)
    plt.title(str(outputs_df.columns[element_nr+1]) + ': '+str(percentage_correct) +'% inside +-' + str(how_many_sd) + ' sd confidence interval', size=25)
    
    if path_to_save is not None:
        plt.savefig(path_to_save + '_mean_plus_minus_sd_intervals_' + str(outputs_df.columns[element_nr+1]))
    plt.show()