
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

if mode == "training" or mode.startswith("benchmark"):
    nrows = 2
    dims_to_keep = "all"
elif mode == "training_subset":
    nrows = 1
    dims_to_keep = [4]

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

# 5-fold cross validation

k = 5

train_loaders, val_loaders, test_loaders, data_lists, original_labels_s = prepare_k_fold_crossvalidation(
                                                                                                            k,
                                                                                                            DATA_PATH,
                                                                                                            how_many_outer_to_remove= HOW_MANY_OUTER_TO_REMOVE,
                                                                                                            elements_to_keep = ELEMENTS_TO_KEEP,
                                                                                                            multiplication_weights = MULTIPLICATION_WEIGHTS,
                                                                                                            preprocessing_method = PREPROCESSING_METHOD, 
                                                                                                            random_state=3,
                                                                                                            normalisation_to_Fe=NORMALISATION_TO_FE,
                                                                                                            return_data=True,
                                                                                                            perturb_data=False,
                                                                                                            mu=0,
                                                                                                            sigma=0.1,
                                                                                                            return_original_labels=True
                                                                                                        )


outputs_rfs, outputs_xgboosts, outputs_surs, outputs_nns, labels_s = [], [], [], [], []

for fold_nr in range(k):

    train_loader, val_loader, test_loader, data_list, original_labels = train_loaders[fold_nr], val_loaders[fold_nr], test_loaders[fold_nr], data_lists[fold_nr], original_labels_s[fold_nr]

    original_labels = get_sample_number_in_group(original_labels)

    X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = data_list

    labels = y_test

    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(
                                        n_estimators=100,
                                        criterion='absolute_error',
                                        random_state=3,
                                        oob_score=True
                                        )
    # Training or loading
    rf_regressor.fit(X_train, y_train)

    # Prediction on test set
    outputs_rf = rf_regressor.predict(X_test)


    # XGBoost regressor    

    xgboost_model = XGBRegressor(
        multi_strategy="multi_output_tree"
    )

    xgboost_model.fit(X_train, y_train)

    outputs_xgboost = xgboost_model.predict(X_test)

    # Surrogate model

    outputs_sur = X_test
    #outputs_sur = np.mean(X_train, axis=0)
    #outputs_sur = np.tile(outputs_sur, (y_test.shape[0], 1))

    # InksNet

    device = get_device()
    model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
    weights = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(device)
    loss_fn = CustomLoss(weights=weights)

    model, train_losses, val_losses = train_model(model=model,
                                                  train_loader=train_loader,
                                                  val_loader=val_loader,
                                                  epochs=2000,
                                                  loss_fn=loss_fn)

    model.eval()
    outputs_nn = model(data_to_device(X_test, device))

    outputs_rfs.append(outputs_rf)
    outputs_xgboosts.append(outputs_xgboost)
    outputs_surs.append(outputs_sur)
    outputs_nns.append(outputs_nn)
    labels_s.append(labels)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_rf.pkl', 'wb') as f:
    pickle.dump(outputs_rfs, f)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_xgboost.pkl', 'wb') as f:
    pickle.dump(outputs_xgboosts, f)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_sur.pkl', 'wb') as f:
    pickle.dump(outputs_surs, f)

with open(RESULTS_PATH["crossvalidation"] + 'outputs_nn.pkl', 'wb') as f:
    pickle.dump(outputs_nns, f)

with open(RESULTS_PATH["crossvalidation"] + 'labels.pkl', 'wb') as f:
    pickle.dump(labels_s, f)
