import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *
from model import *


import json

with open('../config.json', 'r') as f:
    config = json.load(f)

FIGURES_PATH = config['figures_path']['crossvalidation']
RESULTS_PATH = config['results_path']
ELEMENTS_TO_KEEP = config['elements_to_keep']


with open(RESULTS_PATH["crossvalidation"] + 'outputs_rf.pkl', 'rb') as f:
    outputs_rfs = pickle.load(f)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_xgboost.pkl', 'rb') as f:
    outputs_xgboosts = pickle.load(f)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_sur.pkl', 'rb') as f:
    outputs_surs = pickle.load(f)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_nn.pkl', 'rb') as f:
    outputs_nns = pickle.load(f)

with open(RESULTS_PATH["crossvalidation"] + 'labels.pkl', 'rb') as f:
    labels_s = pickle.load(f)


k = 5
summaries_sur, summaries_rf, summaries_xgboost, summaries_nn = [], [], [], []
model_names = ['Surrogate model', 'XGBoost', 'Random Forest', 'InksNet']

for fold_nr in range(k):

    labels = labels_s[fold_nr]

    outputs_sur = outputs_surs[fold_nr]
    outputs_rf = outputs_rfs[fold_nr]
    outputs_xgboost = outputs_xgboosts[fold_nr]
    outputs_nn = outputs_nns[fold_nr]
    
    summary_sur = compute_metrics(outputs_sur, labels)
    summary_rf = compute_metrics(outputs_rf, labels)
    summary_xgboost = compute_metrics(outputs_xgboost, labels)
    summary_nn = compute_metrics(to_numpy(outputs_nn), labels)

    summaries_sur.append(summary_sur)
    summaries_rf.append(summary_rf)
    summaries_xgboost.append(summary_xgboost)
    summaries_nn.append(summary_nn)


# FIGURES FOR EACH OF 5 FOLDS

# for fold_nr in range(k):

#     labels = labels_s[fold_nr]

#     outputs_sur = outputs_surs[fold_nr]
#     outputs_rf = outputs_rfs[fold_nr]
#     outputs_xgboost = outputs_xgboosts[fold_nr]
#     outputs_nn = outputs_nns[fold_nr]
    
#     summary_sur = compute_metrics(outputs_sur, labels)
#     summary_rf = compute_metrics(outputs_rf, labels)
#     summary_xgboost = compute_metrics(outputs_xgboost, labels)
#     summary_nn = compute_metrics(to_numpy(outputs_nn), labels)

#     heatmap_model_metric(
#                             summaries = [summary_sur, summary_xgboost, summary_rf, summary_nn], 
#                             model_names = model_names, 
#                             errors_names = ['mean_mae', 'mean_rmse', 'mean_max_error'],
#                             official_errors_names = ['MAE', 'RMSE', 'Max error'], 
#                             path_to_save=None
#                             )

#     plot_correlation_heatmaps_for_different_models(
#                                                     [to_numpy(outputs_nn), outputs_rf, outputs_xgboost],
#                                                     labels,
#                                                     ['InksNet', 'Random Forest', 'XGBoost'],
#                                                     ELEMENTS_TO_KEEP,
#                                                     xlabel='',
#                                                     ylabel='Residual',
#                                                     cluster=True,
#                                                     path_to_save=None
#                                                 )

#     plot_error_distributions_for_different_models(
#                                                     [outputs_rf, to_numpy(outputs_nn)], 
#                                                     labels, 
#                                                     ['Random Forest', 'InksNet'], 
#                                                     ELEMENTS_TO_KEEP,  
#                                                     dims_to_keep=[4], 
#                                                     nrows=1     
#                                                     )

#     plot_l1_error_with_density(to_numpy(outputs_nn), labels)


# MEAN MAE ACROSS 5 FOLDS FOR EACH MODEL AND ELEMENT

meann = apply_across_folds(
                                [summaries_sur, summaries_rf, summaries_xgboost, summaries_nn],
                                'mae',
                                np.mean,
                                k=5
                                )
stdd = apply_across_folds(
                                [summaries_sur, summaries_rf, summaries_xgboost, summaries_nn],
                                'mae',
                                np.std,
                                k=5
                                )

plot_model_element(
                    errors = meann, 
                    model_names = ['Surrogate model', 'Random forest', 'XGBoost', 'InksNet'], 
                    elements_to_keep = ELEMENTS_TO_KEEP, 
                    figsize=(12, 5), 
                    ylabel='Mean of MAE across folds',
                    path_to_save = FIGURES_PATH + 'mae.png'
                    )

# ZOOMED MEAN MAE ACROSS 5 FOLDS FOR EACH MODEL AND ELEMENT

meann = apply_across_folds(
                                [summaries_rf, summaries_xgboost, summaries_nn],
                                'mae',
                                np.mean,
                                k=5
                                )
stdd = apply_across_folds(
                                [summaries_rf, summaries_xgboost, summaries_nn],
                                'mae',
                                np.std,
                                k=5
                                )

plot_model_element(
                    errors = meann, 
                    model_names = ['Random forest', 'XGBoost', 'InksNet'], 
                    colors = ['gold', 'green', 'darkviolet'],
                    elements_to_keep = ELEMENTS_TO_KEEP, 
                    std = stdd,
                    figsize=(12, 5), 
                    ylabel='Mean of MAE across folds',
                    path_to_save = FIGURES_PATH + 'mae_zoom.png'
                    )

# MEAN MODEL METRIC HEATMAP (ACROSS 5 FOLDS, WITH ANNOTATION)

mean_mae = apply_across_folds(
                                [summaries_sur, summaries_rf, summaries_xgboost, summaries_nn],
                                'mean_mae',
                                np.mean,
                                k=5
                                )
mean_rmse = apply_across_folds(
                                [summaries_sur, summaries_rf, summaries_xgboost, summaries_nn],
                                'mean_rmse',
                                np.mean,
                                k=5
                                )

mean_max_error = apply_across_folds(
                                [summaries_sur, summaries_rf, summaries_xgboost, summaries_nn],
                                'mean_max_error',
                                np.mean,
                                k=5
                                )

summary_5_fold_sur = {}
summary_5_fold_rf = {}
summary_5_fold_xgboost = {}
summary_5_fold_nn = {}
summary_5_fold_sur['mean_mae'] = mean_mae[0]
summary_5_fold_rf['mean_mae'] = mean_mae[1]
summary_5_fold_xgboost['mean_mae'] = mean_mae[2]
summary_5_fold_nn['mean_mae'] = mean_mae[3]
summary_5_fold_sur['mean_rmse'] = mean_rmse[0]
summary_5_fold_rf['mean_rmse'] = mean_rmse[1]
summary_5_fold_xgboost['mean_rmse'] = mean_rmse[2]
summary_5_fold_nn['mean_rmse'] = mean_rmse[3]
summary_5_fold_sur['mean_max_error'] = mean_max_error[0]
summary_5_fold_rf['mean_max_error'] = mean_max_error[1]
summary_5_fold_xgboost['mean_max_error'] = mean_max_error[2]
summary_5_fold_nn['mean_max_error'] = mean_max_error[3]


annot_mean = pd.DataFrame(columns=['MAE', 'RMSE', 'Max error'], index=['Surrogate model', 'XGBoost', 'Random Forest', 'InksNet'], dtype=float)
annot_mean.iloc[:,0] = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_mae',
                                np.mean,
                                k=5
                                )
annot_mean.iloc[:,1] = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_rmse',
                                np.mean,
                                k=5
                                )
annot_mean.iloc[:,2] = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_max_error',
                                np.mean,
                                k=5
                                )

annot_mean.iloc[:,:] = np.round(annot_mean.values, 2)

annot_std = pd.DataFrame(columns=['MAE', 'RMSE', 'Max error'], index=['Surrogate model', 'XGBoost', 'Random Forest', 'InksNet'], dtype=float)
annot_std.iloc[:,0] = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_mae',
                                np.std,
                                k=5
                                )
annot_std.iloc[:,1] = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_rmse',
                                np.std,
                                k=5
                                )
annot_std.iloc[:,2] = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_max_error',
                                np.std,
                                k=5
                                )
annot_std .iloc[:,:]= np.round(annot_std.values, 2)

annot = annot_mean.astype(str) + ' (+- ' + annot_std.astype(str) +')'

heatmap_model_metric(
                    [summary_5_fold_sur, summary_5_fold_xgboost, summary_5_fold_rf, summary_5_fold_nn],
                    ['Surrogate model', 'XGBoost', 'Random Forest', 'InksNet'], 
                    ['mean_mae', 'mean_rmse', 'mean_max_error'],
                     ['MAE', 'RMSE', 'Max error'],
                     annot=annot,
                     fmt='',
                    path_to_save = FIGURES_PATH + 'heatmap_mae.png')


mean_mae = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_l2',
                                np.mean,
                                k=5
                                )
std_mae = apply_across_folds(
                                [summaries_sur, summaries_xgboost, summaries_rf, summaries_nn],
                                'mean_l2',
                                np.std,
                                k=5
                                )

print(mean_mae)
print(std_mae)