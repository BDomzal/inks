import sys
sys.path.insert(1, '../src/')
from data_utils import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

DATASET = "training"

PREPROCESSING_METHOD = config["preprocessing_method"] # normalization / logarithm
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
ELEMENTS_TO_KEEP_NO_FE = config["elements_to_keep_no_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]

PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET.split('_')[0]]

corroded = ['192', '193', '194', '197', '221', '222', '223', '224', '225']

inDKs_df, inks_df, inds_df = load_training_data(PREPROCESSED_DATA_PATH)

# Further preprocessing (-> inks_nn_regression)
inDKs_df = create_sample_id(inDKs_df)

inDKs_df = remove_outer_samples(inDKs_df, HOW_MANY_OUTER_TO_REMOVE)

inDKs_df = delete_elements(inDKs_df, ELEMENTS_TO_KEEP)

inDKs_df = remove_missing_data(inDKs_df)

inDKs_df = set_negative_to_zero(inDKs_df, ELEMENTS_TO_KEEP)

inDKs_df = divide_by_weights(inDKs_df, ELEMENTS_TO_KEEP, suffix='_i', weights=MULTIPLICATION_WEIGHTS)

inDKs_df = normalize_to_Fe(inDKs_df, ELEMENTS_TO_KEEP)


inDKs_df.iloc[:,:-2] = transform_data(inDKs_df.iloc[:,:-2], PREPROCESSING_METHOD)

inds_df, inks_df = split_inDKs_df(inDKs_df)
inds_df.columns = [name.split('_')[0] for name in inds_df.columns]
inks_df.columns = [name.split('_')[0] for name in inks_df.columns]
inds_df['name'] = inDKs_df['name']
inks_df['name'] = inDKs_df['name']


def sample_in_list(sample, list_of_names):
    return any([sample.startswith(el) for el in list_of_names])

corroded_inds_df = inds_df[inds_df['name'].apply(lambda x: sample_in_list(x, corroded))][ELEMENTS_TO_KEEP_NO_FE + ['name']]
corroded_inks_df = inks_df[inks_df['name'].apply(lambda x: sample_in_list(x, corroded))][ELEMENTS_TO_KEEP_NO_FE]

corroded_inds_df.to_csv('../data/corroded/preprocessed_data/corroded_inds.csv', index=False)
corroded_inks_df.to_csv('../results/corroded/corroded_inks.csv', index=False, header=False)