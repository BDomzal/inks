import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *
from model import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

mode = "training"

PREPROCESSING_METHOD = config["preprocessing_method"]
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]
FIGURES_PATH = config["figures_path"][mode]

if mode == "training":
    nrows = 2
    dims_to_keep = "all"
else:
    nrows = 1
    dims_to_keep = [7]

## Loading data and trained model

# Preparing training, validation, test data

train_loader, val_loader, test_loader, data_list, original_labels = prepare_training_data(
                                                                        data_path = DATA_PATH,
                                                                        how_many_outer_to_remove = HOW_MANY_OUTER_TO_REMOVE,
                                                                        elements_to_keep = ELEMENTS_TO_KEEP,
                                                                        multiplication_weights = MULTIPLICATION_WEIGHTS,
                                                                        preprocessing_method = PREPROCESSING_METHOD, 
                                                                        return_data=True,
                                                                        normalisation_to_Fe=NORMALISATION_TO_FE,
                                                                        return_original_labels=True
                                                                        )

original_labels = get_sample_number_in_group(original_labels)

X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = data_list


device = get_device()
X_train = data_to_device(X_train, device)
y_train = data_to_device(y_train, device)
X_val = data_to_device(X_val, device)
y_val = data_to_device(y_val, device)
X_test = data_to_device(X_test, device)
y_test = data_to_device(y_test, device)


# Loading pretrained model

model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
model.load_state_dict(torch.load(MODELS_PATH, weights_only=False))

# Defining loss function

weights = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(device)
loss_fn = CustomLoss(weights=weights)

# Prediction on test set

labels, outputs, difference, mean_loss = evaluate_on_test_set(model=model, 
                                                              test_loader=test_loader, 
                                                              loss_fn=loss_fn)

# First five in group only
# labels = labels[original_labels<=5]
# outputs = outputs[original_labels<=5]
# test_order = test_order[original_labels<=5]

## Validation on test set (never seen by InksNet)

# Consecutive inks from test set are in labels tensor:
print('y_true:')
print(labels)


# Predictions are in outputs tensor:
print('y_pred:')
print(outputs)


# Absolute value of difference between true values (labels) and prediction (outputs) are stored in difference tensor:
print('Difference:')
print(difference)


# Mean loss:
print('Mean loss:')
print(mean_loss)

# Loss on consecutive coordinates:
print('Loss on consecutive coordinates:')
print(torch.mean(difference, axis=0))
# Consecutive coordinates correspond to: Al, S, Cr, Mn, Co, Cu, Zn, Pb, Fe, Mg.
# Significance of elements: Cu >> Mn > Al > Zn > Pb > S > Cr > Co >> all the others.

# BASIC DIAGNOSTIC PLOTS

summary = compute_metrics(to_numpy(outputs), to_numpy(labels))
print(summary)

if mode == "training":

    plot_pred_vs_gt(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Logarithmed true value', ylabel='Logarithmed prediction', path_to_save=FIGURES_PATH)

    plot_residuals(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Logarithmed true value', path_to_save=FIGURES_PATH)

    plot_error_distributions(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Residual', ylabel='Count', path_to_save=FIGURES_PATH)

    plot_qq(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Theoretical quantiles', ylabel='Prediction quantiles', path_to_save=FIGURES_PATH)


if mode == "training_subset":

    plot_pred_vs_gt(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=[1, 3, 4, 7, 9, 10], nrows=nrows, figsize=(14, 5), xlabel='Logarithmed true value', ylabel='Logarithmed prediction', path_to_save=FIGURES_PATH)

    plot_residuals(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, figsize=(8, 5), xlabel='Logarithmed true value', path_to_save=FIGURES_PATH)

    plot_error_distributions(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, figsize=(8, 5), xlabel='Residual', ylabel='Count', path_to_save=FIGURES_PATH)

    plot_qq(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, figsize=(8, 5), xlabel='Theoretical quantiles', ylabel='Prediction quantiles', path_to_save=FIGURES_PATH)

plot_error_boxplot(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep='all', xlabel='', ylabel='Residual', path_to_save=FIGURES_PATH)

plot_error_violinplot(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, dims_to_keep='all', xlabel='', ylabel='Residual', path_to_save=FIGURES_PATH)

plot_correlation_heatmaps(to_numpy(outputs), to_numpy(labels), ELEMENTS_TO_KEEP, xlabel='', ylabel='Residual', cluster=True, path_to_save=FIGURES_PATH)

plot_l1_error(to_numpy(outputs), to_numpy(labels), path_to_save=FIGURES_PATH)

plot_l1_error_with_density(to_numpy(outputs), to_numpy(labels), path_to_save=FIGURES_PATH)
