import sys
sys.path.insert(1, '../src/')
from raw_data_preprocessing import *
from data_utils import *


DATASET = "training"


import matplotlib.pyplot as plt
import json

with open('../config.json', 'r') as f:
    config = json.load(f)

RAW_DATA_PATH = config["raw_data_path"][DATASET]
PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET]
ELEMENTS_DICT = config["elements_dict"]
ELEMENTS_TO_KEEP = config["elements_to_keep"]
CHEMICAL_ELEMENTS_TRANSLATOR = config["chemical_elements_translator"]


inds_df, inks_df = alternative_preprocessing(
                                            RAW_DATA_PATH,
                                            ELEMENTS_DICT,
                                            CHEMICAL_ELEMENTS_TRANSLATOR,
                                            ELEMENTS_TO_KEEP
                                            )


inDKs_df = pd.concat([inks_df, inds_df], axis=1)


inDKs_df.to_csv(PREPROCESSED_DATA_PATH, index=False)
