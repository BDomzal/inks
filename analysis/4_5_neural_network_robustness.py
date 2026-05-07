import sys
sys.path.insert(1, '../src/')
from data_utils import *
from robustness_analysis import *
from train_and_evaluate import *
from model import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

PREPROCESSING_METHOD = config["preprocessing_method"]
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]
FIGURES_PATH = config["figures_path"]["robustness"]

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
                                                                        perturb_data=True,
                                                                        sigma=0.2,
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
labels = to_numpy(labels)
outputs = to_numpy(outputs)
inputs = to_numpy(X_test)

# Error vs input feature

plot_error_vs_input(inputs, outputs, labels, ELEMENTS_TO_KEEP, dims_to_keep='all', nrows=2, figsize=(12,5), xlabel='Input value', ylabel='Error', path_to_save=FIGURES_PATH)


# ICE (Individual Conditional Expectation) profiles

for element_nr in range(len(ELEMENTS_TO_KEEP)):

    grid, lek_profiles, ice_profiles = lek_profile_multi_output(model, X_test, element_nr)

    plot_ice_profiles(
                        grid, 
                        lek_profiles, 
                        ice_profiles, 
                        ELEMENTS_TO_KEEP, 
                        xlabel = 'Input value for ' + ELEMENTS_TO_KEEP[element_nr], 
                        ylabel='Predicted values', 
                        path_to_save=FIGURES_PATH + 'ice_profiles_' + ELEMENTS_TO_KEEP[element_nr]
                        )


# OOD (Out Of Distribution) Samples Analysis

results = get_ood_results(model, X_test, labels, ELEMENTS_TO_KEEP)

#error degradation
perturbation_names = [r["label"] for r in results]
mae_means = [r["mae_mean"] for r in results]
mae_stds = [r["mae_std"] for r in results]
mae_all = np.array([r["mae_all"] for r in results]).T

plot_ood_results(mae_means, mae_stds, perturbation_names, path_to_save=FIGURES_PATH)
plot_ood_results_violin(mae_all, perturbation_names, path_to_save=FIGURES_PATH)