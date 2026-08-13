import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *
from model import *
import time

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
FIGURES_PATH = config["figures_path"]["training"]

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

print("Train target std (per coord):", np.std(y_train, axis=0))
print("Val target std (per coord):", np.std(y_val, axis=0))
print("Train target std (overall):", np.std(y_train))
print("Val target std (overall):", np.std(y_val))

# Building neural network

device = get_device()
model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)

# Setting loss function

weights = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(device)
loss_fn = CustomLoss(weights=weights)
#loss_fn = CustomLoss4()


# Training the neural network 


start = time.time()
model, train_losses, val_losses = train_model(model=model,
                                              train_loader=train_loader,
                                              val_loader=val_loader,
                                              epochs=2000,
                                              loss_fn=loss_fn)
stop=time.time()
print('Training took: ' + str(stop-start) + ' seconds.')
# Loss visualisation

visualise_losses(train_losses, val_losses, path_to_save=FIGURES_PATH)

# Saving the model

save_model(model, MODELS_PATH, HOW_MANY_OUTER_TO_REMOVE, 
                ELEMENTS_TO_KEEP, PREPROCESSING_METHOD)