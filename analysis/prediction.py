import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *


import json

with open('../config.json', 'r') as f:
    config = json.load(f)

DATASET = "Kopernik"

PREPROCESSING_METHOD = config["preprocessing_method"] # normalization / logarithm
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]

MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET]
MODELS_PATH = config["models_path"]
RESULTS_PATH = config["results_path"][DATASET]

ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']


# ## Loading the data

X  =  prepare_target_data(PREPROCESSED_DATA_PATH,
                            ELEMENTS_TO_KEEP,
                            HOW_MANY_OUTER_TO_REMOVE,
                            MULTIPLICATION_WEIGHTS,
                            PREPROCESSING_METHOD, 
                            header=0,
                            column_to_use='name',
                            return_numpy=False
                            )

# ## Loading the trained model

device = get_device()
model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
model.load_state_dict(torch.load(MODELS_PATH, weights_only=False))


# ## Prediction

model.eval()
prediction = model(X)


# ## Saving result to a file

model_name = MODELS_PATH.split('/')[-1]
save_prediction(prediction, RESULTS_PATH, model_name)

