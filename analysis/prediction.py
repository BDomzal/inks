import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *


import json

with open('../config.json', 'r') as f:
    config = json.load(f)

DATASET = "Konstytucja"

PREPROCESSING_METHOD = config["preprocessing_method"] # normalization / logarithm
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]

MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]
CHEMICAL_ELEMENTS_TRANSLATOR = config["chemical_elements_translator"]
REVERSED_CHEMICAL_ELEMENTS_TRANSLATOR = {v: k for k, v in CHEMICAL_ELEMENTS_TRANSLATOR.items()}

PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET]
MODELS_PATH = config["models_path"]
RESULTS_PATH = config["results_path"][DATASET]

ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']


# ## Loading the data


inds_df = load_target_data(PREPROCESSED_DATA_PATH, ELEMENTS_TO_KEEP + ['name'], header=0)


# ## Preprocessing

# 1. In this case, we skip removal of outer samples - it was only done for training examples to ensure the quality of training dataset. No need to do it on target dataset.

# 2. We skip removing columns from outside ELEMENTS_TO_KEEP list, as we have already done it during dataset loading.

# 3. There is no missing data in target dataset, so there is no need to remove anything.


inds_df.dropna().shape == inds_df.shape


# 4. For indicators, we do the same preprocessing as for the training dataset (-> notebooks/inks_nn_regression.ipynb). First, we set negative numbers to zero.


inds_df = set_negative_to_zero_v2(inds_df, ELEMENTS_TO_KEEP)


# 5. Let's divide the indicators by weights.


inds_df = divide_by_weights(inds_df, ELEMENTS_TO_KEEP, suffix='', weights=MULTIPLICATION_WEIGHTS)


# 6. We normalize to Fe.


inds_df = normalize_to_Fe_v2(inds_df, ELEMENTS_TO_KEEP)


# 7. Remove Fe.


inds_df = delete_elements_v2(inds_df, ELEMENTS_TO_KEEP_NO_FE)


# 7. Finally, let's reset the index.


inds_df.reset_index(drop=True, inplace=True)


# ### Converting to np.array

# (Everything except Sample_id column.)


X = np.array(inds_df[ELEMENTS_TO_KEEP_NO_FE].values)


# ### Normalizing / taking logarithm



X = transform_data(X, PREPROCESSING_METHOD)



coor = 5
plt.hist(X[:,coor], bins=100);


# ### Converting to tensors



device = get_device()
X = data_to_device(X, device)


# ## Loading the trained model


model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
model.load_state_dict(torch.load(MODELS_PATH))


# ## Prediction


model.eval()
prediction = model(X)


# ## Saving result to a file


model_name = MODELS_PATH.split('/')[-1]
save_prediction(prediction, RESULTS_PATH, model_name)


# READY_RESULTS_PATH = config["results_path"][DATASET]
# input_data = np.loadtxt(RESULTS_PATH, delimiter=',')
# df = pd.DataFrame(data=input_data, columns=ELEMENTS_TO_KEEP)
# df.to_csv('prediction.csv', index=False)

