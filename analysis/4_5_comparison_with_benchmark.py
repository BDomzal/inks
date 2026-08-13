import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *
from model import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

mode = "benchmark_rf"

PREPROCESSING_METHOD = config["preprocessing_method"]
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]
FIGURES_PATH = config["figures_path"][mode]

if mode == "training" or mode.startswith("benchmark"):
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
                                                                        return_original_labels=True,
                                                                        random_state=3
                                                                        )

original_labels = get_sample_number_in_group(original_labels)


X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = data_list

# Defining model, training and prediction

if mode == "benchmark_rf":

    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(
                                        n_estimators=100,
                                        criterion='absolute_error',
                                        random_state=3,
                                        oob_score=True
                                        )
    # Training or loading
    rf_regressor = load_or_train_model('/'.join(MODELS_PATH.split('/')[:-1]) + '/' + 'rf', rf_regressor, X_train, y_train)

    # Prediction on test set
    outputs = rf_regressor.predict(X_test)


elif mode == "benchmark_xgboost":
    
    xgboost_model = XGBRegressor(
        multi_strategy="multi_output_tree"
    )

    xgboost_model = load_or_train_model('/'.join(MODELS_PATH.split('/')[:-1]) + '/' + 'xgboost', xgboost_model, X_train, y_train)

    outputs = xgboost_model.predict(X_test)

elif mode == "benchmark_sur":

    outputs = X_test
    # outputs = np.mean(X_test, axis=0)
    # outputs = np.tile(outputs, (y_test.shape[0], 1))

labels = y_test
difference = abs(outputs-labels)
mean_loss = np.mean(difference)
print(difference)
print(np.mean(difference))

# # First five in group only
# # labels = labels[original_labels<=5]
# # outputs = outputs[original_labels<=5]
# # test_order = test_order[original_labels<=5]

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


# # Mean loss:
print('Mean loss:')
print(mean_loss)

# # Loss on consecutive coordinates:
print('Loss on consecutive coordinates:')
print(np.mean(difference, axis=0))
# # Consecutive coordinates correspond to: Al, S, Cr, Mn, Co, Cu, Zn, Pb, Fe, Mg.
# # Significance of elements: Cu >> Mn > Al > Zn > Pb > S > Cr > Co >> all the others.

# # BASIC DIAGNOSTIC PLOTS

summary = compute_metrics(outputs, labels)
print(summary)

plot_pred_vs_gt(outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Logarithmed true value', ylabel='Logarithmed prediction', path_to_save=FIGURES_PATH)

plot_residuals(outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Logarithmed true value', path_to_save=FIGURES_PATH)

plot_error_distributions(outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Residual', ylabel='Count', path_to_save=FIGURES_PATH)

plot_qq(outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep=dims_to_keep, nrows=nrows, xlabel='Theoretical quantiles', ylabel='Prediction quantiles', path_to_save=FIGURES_PATH)

plot_error_boxplot(outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep='all', xlabel='', ylabel='Residual', path_to_save=FIGURES_PATH)

plot_error_violinplot(outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep='all', xlabel='', ylabel='Residual', path_to_save=FIGURES_PATH)

plot_correlation_heatmaps(outputs, labels, ELEMENTS_TO_KEEP, xlabel='', ylabel='Residual', cluster=True, path_to_save=FIGURES_PATH)

plot_l1_error(outputs, labels, path_to_save=FIGURES_PATH)

plot_l1_error_with_density(outputs, labels, path_to_save=FIGURES_PATH)

########################################################################
# ALL MODELS

# Surrogate model
outputs_sur = X_test

# Random Forest Regressor
rf_regressor = RandomForestRegressor(
                                    n_estimators=100,
                                    criterion='absolute_error',
                                    random_state=3,
                                    oob_score=True
                                    )
rf_regressor = load_or_train_model('/'.join(MODELS_PATH.split('/')[:-1]) + '/' + 'rf', rf_regressor, X_train, y_train)
outputs_rf = rf_regressor.predict(X_test)

# XGBoost
xgboost_model = XGBRegressor(
    multi_strategy="multi_output_tree"
)
xgboost_model = load_or_train_model('/'.join(MODELS_PATH.split('/')[:-1]) + '/' + 'xgboost', xgboost_model, X_train, y_train)
outputs_xgboost = xgboost_model.predict(X_test)

# InksNet
device = get_device()
model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
model.load_state_dict(torch.load(MODELS_PATH, weights_only=False))
model.eval()
outputs_nn = to_numpy(model(data_to_device(X_test, device)))

