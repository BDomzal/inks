import sys
sys.path.insert(1, '../src/')
from data_utils import *
from train_and_evaluate import *

import json

with open('../config.json', 'r') as f:
    config = json.load(f)

DATASET = "Merkuriusz"

ELEMENTS_TO_KEEP = config["elements_to_keep"]
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']

PREDICTION_PATH = config["prediction_path"][DATASET]
PREDICTION_INPUT_PATH = config["prediction_input_path"][DATASET]
FIGURES_PATH = config["figures_path"][DATASET]

if DATASET == "Konstytucja":
    CMAP = plt.get_cmap('hsv')
else:
    CMAP = plt.get_cmap('tab20')


# ## Loading the data

df = load_prediction(PREDICTION_PATH, ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP)
inds_df = load_target_data(PREDICTION_INPUT_PATH)
y_true = create_sample_id_in_target_data(inds_df, column_to_use='name')['Sample_id']

# ### Converting to np.array

X = np.array(df.values)


# ### Heatmap - datasets separately

visualise_clustering_on_heatmap(X, y_true.values, ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, figures_path=FIGURES_PATH)


# # ### PCA - datasets separately

from sklearn.decomposition import PCA

n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)


visualise_pca(X_pca, y_true, cmap=CMAP, figures_path=FIGURES_PATH + DATASET)

visualise_means_pca(X_pca, y_true, cmap=CMAP, figures_path=FIGURES_PATH + DATASET)


# ### tSNE - datasets separately

from sklearn.manifold import TSNE

n_components = 2
tsne = TSNE(n_components=n_components, random_state=42)
X_tsne = tsne.fit_transform(X)

visualise_pca(X_tsne, y_true, cmap=CMAP, figures_path=FIGURES_PATH + DATASET, figures_name='tsne')


# ### PCA - all points (no proportions!)

ELEMENTS_TO_KEEP = config["elements_to_keep"]
ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']
NORMALISATION_TO_FE = config["normalisation_to_Fe"]

PREDICTION_PATH_DICT = config["prediction_path"]
PREDICTION_INPUT_PATH_DICT = config["prediction_input_path"]
FIGURES_PATH = config["figures_path"]['all']
EXCEL_PREDICTION_PATH = config['excel_prediction_path']

OFFICIAL_NAMES_DICT = config["official_names"]

DATASETS = ['Konstytucja', 'corroded', 'Merkuriusz', 'Kopernik']

CMAP = plt.get_cmap('tab20')

df, y_true = join_datasets_into_one(
                            DATASETS,
                            PREDICTION_PATH_DICT,
                            PREDICTION_INPUT_PATH_DICT,
                            ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP
                            )

X = np.array(df.values)


from sklearn.decomposition import PCA

n_components = 4
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)



print(pca.explained_variance_ratio_)

print(pca.components_)


print(ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP)
# Most correlated
# Zn Cu Co > Mn / Al

print(y_true)

visualise_pca(X_pca, y_true, dimensions=[0,1], whether_sort=False, figures_path=FIGURES_PATH)

#visualise_means_pca(X_pca, y_true, cmap=CMAP, figures_path=FIGURES_PATH + 'pca_means')

visualise_pca(X_pca, y_true, dimensions=[1, 2], whether_sort=False, figures_path=FIGURES_PATH)


from sklearn.manifold import TSNE

n_components = 2
tsne = TSNE(n_components=n_components, random_state=42)
X_tsne = tsne.fit_transform(X)


visualise_pca(X_tsne, y_true, dimensions=[0, 1], whether_sort=False, figures_path=FIGURES_PATH, figures_name='tsne')


# dfs = load_prediction_list(DATASETS,
#                            PREDICTION_PATH_DICT,
#                            PREDICTION_INPUT_PATH_DICT,
#                            ELEMENTS_TO_KEEP,
#                            NORMALISATION_TO_FE,
#                            logarithm=True)

# save_df_list_to_excel(dfs, EXCEL_PREDICTION_PATH)


# Figure 3 in paper

# PCA-like plot: selected elements

df, y_true = join_datasets_into_one(
                            DATASETS,
                            PREDICTION_PATH_DICT,
                            PREDICTION_INPUT_PATH_DICT,
                            ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP,
                            short_name=False
                            )

df = normalise_to_total(df)
df, y_true = groupby_and_get_mean_of_first_n(df, y_true, n=10)

y_true = truncate_names(y_true)
y_true = translate_names(y_true, OFFICIAL_NAMES_DICT)
X = np.array(df.values)

visualise_selected_axis(
                        X, 
                        y_true,
                        ELEMENTS_TO_KEEP,
                        cmap=plt.get_cmap('Paired'),
                        dimensions_horizontal = [1, 4, 7],
                        dimensions_vertical = [8, 9, 10],
                        path_to_save=FIGURES_PATH
                        )

# Heatmap with clustering

df, y_true = join_datasets_into_one(
                            DATASETS,
                            PREDICTION_PATH_DICT,
                            PREDICTION_INPUT_PATH_DICT,
                            ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP,
                            short_name=False
                            )

df = normalise_to_total(df)
df, y_true = groupby_and_get_mean_of_first_n(df, y_true, n=10)

y_true = truncate_names(y_true)
y_true = translate_names(y_true, OFFICIAL_NAMES_DICT)
X = np.array(df.values)

visualise_clustering_on_heatmap(X, y_true.values, ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, colormap=cm.Paired, dendrogram_ratio=0.2, col_cluster=False, show_legend=False, figures_path=FIGURES_PATH)

# PCA plot

df, y_true = join_datasets_into_one(
                            DATASETS,
                            PREDICTION_PATH_DICT,
                            PREDICTION_INPUT_PATH_DICT,
                            ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP,
                            short_name=False
                            )

df = normalise_to_total(df)
df, y_true = groupby_and_get_mean_of_first_n(df, y_true, n=10)

y_true = truncate_names(y_true)
y_true = translate_names(y_true, OFFICIAL_NAMES_DICT)
X = np.array(df.values)

X = np.array(df.values)


n_components = 4
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)


print(pca.explained_variance_ratio_)

print(pca.components_)


print(ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP)
# Most correlated
# Zn Cu Co > Mn / Al

print(y_true)

visualise_selected_axis(
                        X_pca, 
                        y_true,
                        ['PC1', 'PC2', 'PC3', 'PC4'],
                        cmap=plt.get_cmap('Paired'),
                        dimensions_horizontal = [0],
                        dimensions_vertical = [1],
                        path_to_save=FIGURES_PATH + 'pca_'
                        )
