import sys
sys.path.insert(1, '../src/')
from raw_data_preprocessing import *
from data_utils import *


DATASET = "artificial_inks"


import matplotlib.pyplot as plt
import json

with open('../config.json', 'r') as f:
    config = json.load(f)

RAW_DATA_PATH = config["raw_data_path"][DATASET]
PREPROCESSED_DATA_PATH = config["preprocessed_data_path"][DATASET]
ELEMENTS_DICT = config["elements_dict"]
N_STD = config["n_std"]
ROW_NR = config["row_nr"]
N_DES = config["n_des"]
N = config["n"]
BUNCH_NO = config["bunch_no"]
TO_FE = config["to_fe"]
KEEP_FE = config["keep_fe"]


preprocessing_results = preprocess_all_from_directory(
                                                        raw_data_path = RAW_DATA_PATH, 
                                                        elements_dict = ELEMENTS_DICT, 
                                                        preprocessed_data_path = PREPROCESSED_DATA_PATH,
                                                        n_std = N_STD, 
                                                        row_nr = ROW_NR, 
                                                        n_des = N_DES, 
                                                        n = N, 
                                                        bunch_no = BUNCH_NO, 
                                                        to_fe = TO_FE, 
                                                        keep_fe = KEEP_FE,
                                                        inks_present = True if (DATASET == "training" or DATASET == 'corroded' or DATASET == 'artificial_inks') else False,
                                                        sort=True
                                                    )

if DATASET == 'corroded' or DATASET == 'artificial_inks':

    preprocessing_results2 = preprocess_all_from_directory(
                                                        raw_data_path = RAW_DATA_PATH, 
                                                        elements_dict = ELEMENTS_DICT, 
                                                        preprocessed_data_path = '',
                                                        n_std = N_STD, 
                                                        row_nr = ROW_NR, 
                                                        n_des = N_DES, 
                                                        n = N, 
                                                        bunch_no = BUNCH_NO, 
                                                        to_fe = TO_FE, 
                                                        keep_fe = KEEP_FE,
                                                        inks_present = True if (DATASET == "training" or DATASET == 'corroded' or DATASET == 'artificial_inks') else False,
                                                        sort=True
                                                    )

    inds_df, inks_df = split_inDKs_df(preprocessing_results2, indicators_suffix='_i', inks_suffix='_a', standardise_names=True)
    inds_df['name'] = inds_df['name'].apply(lambda x: x.replace('.i', ''))
    inds_df.to_csv('/'.join(PREPROCESSED_DATA_PATH.split('/')[:-1]) + '/preprocessed_inds.csv', index=False)
