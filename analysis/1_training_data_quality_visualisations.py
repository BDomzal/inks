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

ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']

# Preparing training, validation, test data

train_loader, val_loader, test_loader, data_list = prepare_training_data(
                                                                                data_path = DATA_PATH,
                                                                                how_many_outer_to_remove = HOW_MANY_OUTER_TO_REMOVE,
                                                                                elements_to_keep = ELEMENTS_TO_KEEP,
                                                                                multiplication_weights = MULTIPLICATION_WEIGHTS,
                                                                                preprocessing_method = PREPROCESSING_METHOD, 
                                                                                return_data=True
                                                                                )

X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = data_list

# Visualisations of distribution

visualise_train_val_test_distributions(X_train, X_val, X_test,
                                        ELEMENTS_TO_KEEP_NO_FE, 
                                        title='Indicators',
                                        logarithmed=True,
                                        horizontal_axis_name = 'Logarithm of relative intensity',
                                        path_to_save=FIGURES_PATH + 'indicators_train_val_test_histograms.png'
                                        )



visualise_train_val_test_distributions(y_train, y_val, y_test,
                                        ELEMENTS_TO_KEEP_NO_FE, 
                                        title='Inks',
                                        logarithmed=True,
                                        horizontal_axis_name = 'Logarithm of relative intensity',
                                        path_to_save=FIGURES_PATH + 'inks_train_val_test_histograms.png'
                                        )


# Zoom on single element

element_nr = 3
plt.hist(X_train[:,element_nr], bins=100, label='training');
plt.hist(X_val[:,element_nr], bins=100, label='validation');
plt.hist(X_test[:,element_nr], bins=100, label='test');
plt.title(ELEMENTS_TO_KEEP[element_nr])
plt.legend()
plt.show()

# Visualisation of means of classes (using PCA)

inks_df = pd.DataFrame(
                        data=np.concatenate([X_train, X_val, X_test]), 
                        columns=ELEMENTS_TO_KEEP_NO_FE
                        )

order = pd.concat([train_order, val_order, test_order], axis=0).reset_index(drop=True)


mean_df = create_means_df(
                        ELEMENTS_TO_KEEP_NO_FE, 
                        inks_df.values, 
                        order, 
                        np.array([]), 
                        pd.Series([])
                        )


# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(inks_df.values)
means_pca = pca.transform(mean_df[ELEMENTS_TO_KEEP_NO_FE])

# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(inks_df.values)
# means_tsne does not exist, because there is no tsne.transform method!


visualise_pca_and_class_means(X_pca, order, 
                                means_pca, mean_df['Sample_id'],
                                # x_lower=-1, x_upper=1,
                                # y_lower=2, y_upper=3,
                                cmap = plt.get_cmap('tab20'),
                                seed=14,
                                path_to_save=FIGURES_PATH + 'PCA_and_class_centers.png')

