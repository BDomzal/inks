import sys
sys.path.insert(1, '../src/')
from data_utils import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

#DATASET = "training"
#DATASET = "artificial_inks"
DATASET = "corroded"

PREPROCESSING_METHOD = config["preprocessing_method"] # normalization / logarithm
HOW_MANY_OUTER_TO_REMOVE = config["how_many_outer_to_remove"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
CORRODED = config["corroded"]

PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET]
INKS_ONLY_PATH = config["inks_only_path"][DATASET]

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
    if DATASET == "corroded":
        inds_df = inds_df[inds_df['name'].apply(lambda x: sample_in_list(x, CORRODED))]
        inks_df = inks_df[inks_df['name'].apply(lambda x: sample_in_list(x, CORRODED))]
    inds_df = inds_df[ELEMENTS_TO_KEEP_NO_FE + ['name']]
    inks_df = inks_df[ELEMENTS_TO_KEEP_NO_FE]
else:
    if DATASET == "corroded":
        inds_df = inds_df[inds_df['name'].apply(lambda x: sample_in_list(x, CORRODED))]
        inks_df = inks_df[inks_df['name'].apply(lambda x: sample_in_list(x, CORRODED))]
    inds_df = inds_df[ELEMENTS_TO_KEEP + ['name']]
    inks_df = inks_df[ELEMENTS_TO_KEEP]

inks_df.to_csv(INKS_ONLY_PATH, index=False, header=False)