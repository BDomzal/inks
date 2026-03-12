import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np

import sys
sys.path.insert(1, '../src/')
from data_utils import *
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

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]

# Preparing training, validation, test data

train_loader, val_loader, test_loader, data_list = prepare_training_data(
                                                                        data_path = DATA_PATH,
                                                                        how_many_outer_to_remove = HOW_MANY_OUTER_TO_REMOVE,
                                                                        elements_to_keep = ELEMENTS_TO_KEEP,
                                                                        multiplication_weights = MULTIPLICATION_WEIGHTS,
                                                                        preprocessing_method = PREPROCESSING_METHOD, 
                                                                        return_data=True,
                                                                        normalisation_to_Fe=NORMALISATION_TO_FE
                                                                        )

X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = data_list

device = get_device()

def objective(trial):

    model = build_model(trial)
    model = model.to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    model, train_losses, val_losses = train_model_optuna(model=model,
                                              train_loader=train_loader,
                                              val_loader=val_loader,
                                              epochs=1000,
                                              loss_fn=loss_fn, 
                                              optimizer=optimizer)

    labels, outputs, difference, mean_loss = evaluate_on_test_set(model=model, 
                                                              test_loader=test_loader, 
                                                              loss_fn=loss_fn)

    return mean_loss

# Setting loss function

weights = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(device)
loss_fn = CustomLoss(weights=weights)
#loss_fn = CustomLoss4()

# -----------------------------
# Run optimization
# -----------------------------
study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=100)


print("Best trial:")
print(study.best_trial.params)
print("Best validation loss:", study.best_value)