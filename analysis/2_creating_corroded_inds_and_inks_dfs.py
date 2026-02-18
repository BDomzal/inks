import sys
sys.path.insert(1, '../src/')
from data_utils import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

DATASET = "training"

PREPROCESSING_METHOD = config["preprocessing_method"] # normalization / logarithm
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
CORRODED = config["corroded"]

PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET.split('_')[0]]


inds_df, inks_df, sample_order = prepare_data_without_splitting(
                                                                PREPROCESSED_DATA_PATH,
                                                                HOW_MANY_OUTER_TO_REMOVE,
                                                                ELEMENTS_TO_KEEP,
                                                                MULTIPLICATION_WEIGHTS,
                                                                PREPROCESSING_METHOD,
                                                                indicators_suffix='_i', 
                                                                inks_suffix='_a',
                                                                normalisation_to_Fe=NORMALISATION_TO_FE
                                                                )

if NORMALISATION_TO_FE:
    ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']
    corroded_inds_df = inds_df[inds_df['name'].apply(lambda x: sample_in_list(x, CORRODED))][ELEMENTS_TO_KEEP_NO_FE + ['name']]
    corroded_inks_df = inks_df[inks_df['name'].apply(lambda x: sample_in_list(x, CORRODED))][ELEMENTS_TO_KEEP_NO_FE]
else:
    corroded_inds_df = inds_df[inds_df['name'].apply(lambda x: sample_in_list(x, CORRODED))][ELEMENTS_TO_KEEP + ['name']]
    corroded_inks_df = inks_df[inks_df['name'].apply(lambda x: sample_in_list(x, CORRODED))][ELEMENTS_TO_KEEP]

corroded_inds_df.to_csv('../data/corroded/nn_ready/corroded_inds.csv', index=False)
corroded_inks_df.to_csv('../results/corroded/corroded_inks.csv', index=False, header=False)