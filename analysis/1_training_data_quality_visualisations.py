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
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

DATA_PATH = config["training_data_path"]
FIGURES_PATH = config["figures_path"]["training"]

# Preparing training, validation, test data

inDKs_df = prepare_training_data_for_visualisations(
                                                    data_path = DATA_PATH,
                                                    how_many_outer_to_remove = HOW_MANY_OUTER_TO_REMOVE,
                                                    elements_to_keep = ELEMENTS_TO_KEEP,
                                                    multiplication_weights = MULTIPLICATION_WEIGHTS,
                                                    preprocessing_method = PREPROCESSING_METHOD, 
                                                    return_data=True
                                                    )

inds_df, inks_df = split_inDKs_df(inDKs_df)

ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']
mean_df = create_means_df(ELEMENTS_TO_KEEP, inks_df.values, inDKs_df['Sample_id'], np.array([]), pd.Series([]))


# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

inds_df, inks_df = split_inDKs_df(inDKs_df)

X_pca = pca.fit_transform(inks_df.values)
means_pca = pca.transform(mean_df[ELEMENTS_TO_KEEP])

# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(inks_df.values)
# means_tsne does not exist, because there is no tsne.transform method!


visualise_pca_and_class_means(X_pca, inDKs_df['Sample_id'], 
                                means_pca, mean_df['Sample_id'],
                                x_lower=-4.08, x_upper=0.92,
                                y_lower=2.9, y_upper=7.9,
                                cmap = plt.get_cmap('hsv'),
                                seed=14,
                                path_to_save=None)
                                #path_to_save=FIGURES_PATH + 'PCA_and_class_centers_zoom_14.png') #421 2 10 14



X_y_train, X_y_val, X_y_test = create_partition(inDKs_df)


X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = split_to_X_and_y(X_y_train, 
                                                                                                       X_y_val, 
                                                                                                       X_y_test, 
                                                                                                       ELEMENTS_TO_KEEP)

visualise_train_val_test_distributions(X_train, X_val, X_test,
                                        ELEMENTS_TO_KEEP_NO_FE, 
                                        title='Indicators: ',
                                        path_to_save=None
                                        #path_to_save=FIGURES_PATH + 'indicators_train_val_test_histograms.png'
                                        )



visualise_train_val_test_distributions(y_train, y_val, y_test,
                                        ELEMENTS_TO_KEEP_NO_FE, 
                                        title='Inks: ',
                                        path_to_save=None
                                        #path_to_save=FIGURES_PATH + 'inks_train_val_test_histograms.png'
                                        )


# # Zoom on single element

# dim = 3
# plt.hist(X_train[:,dim], bins=100, label='training');
# plt.hist(X_val[:,dim], bins=100, label='validation');
# plt.hist(X_test[:,dim], bins=100, label='test');
# plt.title(ELEMENTS_TO_KEEP[dim])
# plt.legend()

