
import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *
from model import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

mode = "training_subset"

PREPROCESSING_METHOD = config["preprocessing_method"]
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]
FIGURES_PATH = config["figures_path"][mode]
RESULTS_PATH = config["results_path"]


nrows = 2
dims_to_keep = "all"


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

labels = y_test


########################################################################

# ALL MODELS

model_names = ['Surrogate model',  'XGBoost', 'Random Forest','InksNet']
mean_errors_names = ["mean_mae", "mean_rmse", "mean_l2"]

# Surrogate model
outputs_sur = X_test
# outputs_sur = np.mean(X_test, axis=0)
# outputs_sur = np.tile(outputs_sur, (y_test.shape[0], 1))

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

# Figures comparing models

summary_sur = compute_metrics(outputs_sur, labels)
summary_rf = compute_metrics(outputs_rf, labels)
summary_xgboost = compute_metrics(outputs_xgboost, labels)
summary_nn = compute_metrics(outputs_nn, labels)

mae_sur = summary_sur['mae']
mae_rf = summary_rf['mae']
mae_xgboost = summary_xgboost['mae']
mae_nn = summary_nn['mae']

rmse_sur = summary_sur['rmse']
rmse_rf = summary_rf['rmse']
rmse_xgboost = summary_xgboost['rmse']
rmse_nn = summary_nn['rmse']

r2_sur = summary_sur['r2']
r2_rf = summary_rf['r2']
r2_xgboost = summary_xgboost['r2']
r2_nn = summary_nn['r2']

l2_sur = summary_sur['l2']
l2_rf = summary_rf['l2']
l2_xgboost = summary_xgboost['l2']
l2_nn = summary_nn['l2']

max_sur = summary_sur['max_error']
max_rf = summary_rf['max_error']
max_xgboost = summary_xgboost['max_error']
max_nn = summary_nn['max_error']

bias_sur = summary_sur['bias']
bias_rf = summary_rf['bias']
bias_xgboost = summary_xgboost['bias']
bias_nn = summary_nn['bias']


# Plots model element

plot_model_element(
                    errors = [mae_sur, mae_rf, mae_xgboost, mae_nn], 
                    model_names = model_names, 
                    elements_to_keep = ELEMENTS_TO_KEEP, 
                    figsize=(12, 5), 
                    ylabel='MAE', 
                    # y_lower_lim=0.45,
                    # y_upper_lim=1.75,
                    path_to_save=config["figures_path"]['all'] + 'mae.png'
                    )

plot_model_element(
                    errors = [rmse_sur, rmse_rf, rmse_xgboost, rmse_nn], 
                    model_names = model_names, 
                    elements_to_keep = ELEMENTS_TO_KEEP, 
                    figsize=(12, 5), 
                    ylabel='RMSE', 
                    # y_lower_lim=0.5,
                    # y_upper_lim=2,
                    path_to_save=config["figures_path"]['all'] + 'rmse.png'
                    )

plot_model_element(
                    errors = [r2_sur, r2_rf, r2_xgboost, r2_nn],
                    model_names = model_names,
                    elements_to_keep = ELEMENTS_TO_KEEP,
                    figsize=(12, 5),
                    ylabel='R$^2$',
                    # y_lower_lim=-1,
                    # y_upper_lim=1,
                    path_to_save=config["figures_path"]['all'] + 'r2.png'
                    )

plot_model_element(
                    errors = [l2_sur, l2_rf, l2_xgboost, l2_nn],
                    model_names = model_names,
                    elements_to_keep = ELEMENTS_TO_KEEP,
                    figsize=(12, 5),
                    ylabel='L2',
                    # y_lower_lim=-1,
                    # y_upper_lim=1,
                    path_to_save=config["figures_path"]['all'] + 'l2.png'
                    )

plot_model_element(
                    errors = [max_sur, max_rf, max_xgboost, max_nn],
                    model_names = model_names,
                    elements_to_keep = ELEMENTS_TO_KEEP,
                    figsize=(12, 5),
                    ylabel='Max error',
                    # y_lower_lim=-1,
                    # y_upper_lim=1,
                    path_to_save=config["figures_path"]['all'] + 'max.png'
                    )

plot_model_element(
                    errors = [bias_sur, bias_rf, bias_xgboost, bias_nn], 
                    model_names = model_names, 
                    elements_to_keep = ELEMENTS_TO_KEEP, 
                    figsize=(12, 5), 
                    ylabel='Bias', 
                    # y_lower_lim=-1,
                    # y_upper_lim=1,
                    path_to_save=config["figures_path"]['all'] + 'bias.png'
                    )

# Heatmaps model element

heatmap_model_element(
                        errors = [mae_sur, mae_rf, mae_xgboost, mae_nn],
                        model_names = model_names,
                        elements_to_keep = ELEMENTS_TO_KEEP,
                        path_to_save = config["figures_path"]['all'] + 'mae_heatmap.png'
                    )

heatmap_model_element(
                        errors = [rmse_sur, rmse_rf, rmse_xgboost, rmse_nn],
                        model_names = model_names,
                        elements_to_keep = ELEMENTS_TO_KEEP,
                        path_to_save = config["figures_path"]['all'] + 'rmse_heatmap.png'
                    )

heatmap_model_element(
                        errors = [r2_sur, r2_rf, r2_xgboost, r2_nn],
                        model_names = model_names,
                        elements_to_keep = ELEMENTS_TO_KEEP,
                        path_to_save = config["figures_path"]['all'] + 'r2_heatmap.png'
                    )

heatmap_model_element(
                        errors = [l2_sur, l2_rf, l2_xgboost, l2_nn],
                        model_names = model_names,
                        elements_to_keep = ELEMENTS_TO_KEEP,
                        path_to_save = config["figures_path"]['all'] + 'l2_heatmap.png'
                    )

heatmap_model_element(
                        errors = [max_sur, max_rf, max_xgboost, max_nn],
                        model_names = model_names,
                        elements_to_keep = ELEMENTS_TO_KEEP,
                        path_to_save = config["figures_path"]['all'] + 'max_heatmap.png'
                    )

# # Plots model metric

plot_model_metric(
                        summaries = [summary_sur, summary_rf, summary_xgboost, summary_nn], 
                        model_names = model_names, 
                        errors_names = ['mean_mae', 'mean_rmse', 'mean_max_error'],
                        official_errors_names = ['MAE', 'RMSE', 'Max error'], 
                        path_to_save=config["figures_path"]['all'] + 'error_plot.png'
                        )


# Heatmaps model metric


heatmap_model_metric(
                        summaries = [summary_sur, summary_rf, summary_xgboost, summary_nn], 
                        model_names = model_names, 
                        errors_names = ['mean_mae', 'mean_rmse', 'mean_max_error'],
                        official_errors_names = ['MAE', 'RMSE', 'Max error'], 
                        path_to_save=config["figures_path"]['all'] + 'error_heatmap.png'
                        )


# Residual distributions


plot_error_distributions_for_different_models(
                                                [outputs_rf, outputs_nn], 
                                                labels, 
                                                ['Random Forest', 'InksNet'], 
                                                ELEMENTS_TO_KEEP,  
                                                dims_to_keep=dims_to_keep, 
                                                nrows=nrows, 
                                                path_to_save=config["figures_path"]['all'] + 'all_elements_'
                                                )


plot_error_distributions_for_different_models(
                                                [outputs_rf, outputs_nn], 
                                                labels, 
                                                ['Random Forest', 'InksNet'], 
                                                ELEMENTS_TO_KEEP,  
                                                dims_to_keep=[4], 
                                                nrows=1, 
                                                path_to_save=config["figures_path"]['all']
                                                )

plot_correlation_heatmaps_for_different_models(
                                                    [outputs_nn, outputs_rf, outputs_xgboost],
                                                    labels,
                                                    ['InksNet', 'Random Forest', 'XGBoost'],
                                                    ELEMENTS_TO_KEEP,
                                                    xlabel='',
                                                    ylabel='Residual',
                                                    cluster=True,
                                                    path_to_save=config["figures_path"]["all"]
                                                )
