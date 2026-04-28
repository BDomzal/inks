import numpy as np
import matplotlib.pyplot as plt
from data_utils import *
from scipy.spatial.distance import cdist
import torch

# Lek's and ICE profiles
def lek_profile_multi_output(
						    model,
						    X,
						    feature_idx,
						    quantiles=[0.0, 0.25, 0.5, 0.75, 1.0],
						    n_points=100,
						    ice_samples=30,
						    random_state=42
							):

    model.eval()

    rng = np.random.default_rng(random_state)
    X = to_numpy(X)

    device = get_device()
    Xt = torch.Tensor(X).to(device)

    n_outputs = model(Xt[:1]).shape[1]

    # Grid for selected feature
    x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
    grid = np.linspace(x_min, x_max, n_points)

    # --- LEK PROFILES (quantile values) ---
    q_values = np.quantile(X, quantiles, axis=0)
    lek_profiles = []

    for qv in q_values:
        X_temp = np.tile(qv, (n_points, 1))
        X_temp[:, feature_idx] = grid

        Xt_temp = torch.Tensor(X_temp).to(device)
        preds = model(Xt_temp)  # (n_points, n_outputs)
        lek_profiles.append(to_numpy(preds))

    # --- ICE CURVES ---
    idx = rng.choice(len(X), size=min(ice_samples, len(X)), replace=False)
    ice_profiles = []

    for i in idx:
        base = X[i].copy()
        X_temp = np.tile(base, (n_points, 1))
        X_temp[:, feature_idx] = grid

        Xt_temp = torch.Tensor(X_temp).to(device)
        preds = model(Xt_temp)
        ice_profiles.append(to_numpy(preds))

    return grid, np.array(lek_profiles), np.array(ice_profiles)



def plot_ice_profiles(
				    grid,
				    lek_profiles,
				    ice_profiles,
				    elements_to_keep,
				    xlabel='', 
				    ylabel='',
				    quantiles=[0.0, 0.25, 0.5, 0.75, 1.0],
				    path_to_save=None
				  ):

    fig, ax = plt.subplots(2, 6, figsize=(16,5), sharey=True, sharex=True)

    ax = ax.flatten()
    # ICE curves (light gray)
    for output_idx in range(ice_profiles.shape[2]):

	    for i, ice in zip(range(len(ice_profiles)), ice_profiles):

	        #ax[output_idx].plot(grid, ice[:, output_idx], alpha=0.3, label=str(quantiles[i]))
	        ax[output_idx].plot(grid, ice[:, output_idx], alpha=0.3, color='grey')
	        plt.legend()
	        ax[output_idx].set_title(elements_to_keep[output_idx])

	    ax[output_idx].tick_params(axis='both', labelsize=5)


    ax[-1].set_axis_off()

    fig.text(0.52, 0.0, xlabel, ha='center', size=25)
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', size=25)

    handles, labels = ax[-2].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.85, 0.2))
    plt.grid(True)


    if path_to_save:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Out Of Distribution samples

def compute_errors(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=1)   # per sample
    mse = np.mean((y_true - y_pred) ** 2, axis=1)
    return mae, mse

def scale_shift(X, scale_factors):
    return X * scale_factors

def add_noise(X, noise_std):
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def permute_features(X, feature_idx):
    X_perm = X.clone().detach()
    np.random.shuffle(X_perm[:, feature_idx])
    return X_perm


def evaluate_shift(model, X, y_true, label):
    y_pred = model(X)
    y_pred = to_numpy(y_pred)
    mae, _ = compute_errors(y_true, y_pred)

    return {
        "label": label,
        "mae_mean": mae.mean(),
        "mae_std": mae.std(),
        "mae_all": mae
    }

def get_ood_results(model, X, labels, elements_to_keep):

	results = []

	# baseline
	results.append(evaluate_shift(model, X, labels, "original input"))

	# noise levels
	for noise in [0.01, 0.05, 0.1, 0.2]:
	    X_noisy = add_noise(to_numpy(X), noise)
	    results.append(evaluate_shift(model, data_to_device(X_noisy, get_device()), labels, f"noise with std={noise}"))

	# scaling
	for scale in [0.8, 1.2, 1.5]:
	    X_scaled = X * scale
	    results.append(evaluate_shift(model, data_to_device(X_scaled, get_device()), labels, f"scaling times {scale}"))

	# permutation
	for el_nr, el in enumerate(elements_to_keep):
		X_perm = permute_features(X, feature_idx=el_nr)
		results.append(evaluate_shift(model, data_to_device(X_perm, get_device()), labels, "permutating " + el))

	return results

def plot_ood_results(mae_means, mae_stds, perturbation_names, path_to_save=None):

	fig = plt.figure(figsize=(12, 5))
	plt.errorbar(perturbation_names, mae_means, yerr=mae_stds, fmt='o')
	fig.text(0.07, 0.5, 'MAE', va='center', rotation='vertical', size=25)
	plt.xticks(rotation=45)
	plt.grid(True)

	if path_to_save:
	    plt.savefig(path_to_save+'ood_results.png', dpi=300, bbox_inches="tight")
	plt.show()
	plt.close(fig)


def plot_ood_results_violin(mae_all, perturbation_names, path_to_save=None):

	fig, ax = plt.subplots(figsize=(12,5))
	sns.violinplot(data=pd.DataFrame(mae_all, columns=perturbation_names))
	fig.text(0.07, 0.5, 'MAE', va='center', rotation='vertical', size=25)
	ax.set_xticklabels(perturbation_names, rotation=45)
	ax.tick_params(axis='y', labelsize=5)

	if path_to_save:
	    plt.savefig(path_to_save+'ood_results_violin.png', dpi=300, bbox_inches="tight")
	plt.show()
	plt.close(fig)

# Error vs input features

def plot_error_vs_input(inputs, outputs, labels, elements_to_keep, dims_to_keep='all', nrows=2, figsize=(12,5), xlabel='Input value', ylabel='Error', path_to_save=None):

    if dims_to_keep == 'all':
        dims_to_keep = list(range(len(elements_to_keep)))

    n_dims = len(dims_to_keep)
    in_dims_to_keep = [i in dims_to_keep for i in np.array(range(outputs.shape[1]))]
    labels = labels[:, in_dims_to_keep]
    outputs = outputs[:, in_dims_to_keep]
    elements_to_keep = [elements_to_keep[i] for i in dims_to_keep]

    fig, axes = plt.subplots(nrows, n_dims//nrows + 1 if nrows==2 else n_dims//nrows, figsize=figsize, sharey=True)
    if len(dims_to_keep)>1:
        axes = axes.flatten()

    y_max_max = (outputs-labels).max().max()

    for i in range(n_dims):

        if len(dims_to_keep)>1:
            ax = axes[i]
        else:
            ax = axes

        y_true = labels[:, i]
        y_pred = outputs[:, i]
        x_true = inputs[:, i]

        # min_val = y_true.min()
        # max_val = y_true.max()
        # mean_res = np.mean(y_pred-y_true)

        # ax.plot([min_val, max_val], [mean_res, mean_res], 'black')

        ax.scatter(x_true, y_pred-y_true, alpha=0.25)

        title = elements_to_keep[i]

        # y_max = (y_pred-y_true).max()
        # y_min = (y_pred-y_true).min()

        ax.tick_params(axis='both', labelsize=5)
        ax.set_title(title, size=10)

    if nrows==2:
        axes[-1].set_axis_off()

    fig.text(0.5, 0.0, xlabel, ha='center', size=25)
    fig.text(0.06, 0.5, ylabel, va='center', rotation='vertical', size=25)

    #plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save+'error_vs_input.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)