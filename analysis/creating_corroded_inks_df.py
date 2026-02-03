import sys
sys.path.insert(1, '../src/')
from data_utils import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

DATASET = "training"

ELEMENTS_TO_KEEP = config["elements_to_keep"]
ELEMENTS_TO_KEEP_NO_FE = config["elements_to_keep_no_Fe"]
PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET.split('_')[0]]

corroded = ['192', '193', '194', '197', '221', '222', '223', '224', '225']

inDKs_df, inks_df, inds_df = load_training_data(PREPROCESSED_DATA_PATH)

def sample_in_list(sample, list_of_names):
    return any([sample.startswith(el) for el in list_of_names])

corroded_df = inks_df[inks_df['name'].apply(lambda x: sample_in_list(x, corroded))][ELEMENTS_TO_KEEP_NO_FE]

corroded_df.to_csv('../results/corroded/corroded_inks.csv', index=False, header=False)