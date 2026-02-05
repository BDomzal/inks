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
ELEMENTS_TO_KEEP_NO_FE = config["elements_to_keep_no_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]
FIGURES_PATH = config["figures_path"]["training"]

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]



inDKs_df, inks_df, inds_df = load_training_data(DATA_PATH)


# ## Preprocessing

# 0. Keeping track of the records from the same sample

# We will keep this info in 'Sample_id' and 'name' columns.

inDKs_df = create_sample_id_in_training_data(inDKs_df)


# 1. Let's remove 'outer' samples:

inDKs_df = remove_outer_samples(inDKs_df, HOW_MANY_OUTER_TO_REMOVE)


# 2. Let's keep only columns that we need.

# To reduce the set of used elements run cell below. Then, instead of predicting 29 numbers, we will predict only 10. We will also use only 10 numbers as input.

inDKs_df = delete_elements(inDKs_df, ELEMENTS_TO_KEEP)


# 3. Let's remove rows with missing values if there are any.

inDKs_df = remove_missing_data(inDKs_df)


# 4. Let's set negative numbers to zeros. (First, check if there are any.)

inDKs_df = set_negative_to_zero(inDKs_df)


# 5. Let's divide the indicators by weights (leave inks as they are!)

inDKs_df = divide_by_weights(inDKs_df, ELEMENTS_TO_KEEP, suffix='_i', weights=MULTIPLICATION_WEIGHTS)

# 6. Normalize with respect to Fe and remove Fe (both indicators and inks).

inDKs_df = normalize_to_Fe(inDKs_df, ELEMENTS_TO_KEEP, suffixes=['_i', '_a'])

# 7. Remove Fe.

inDKs_df = delete_elements(inDKs_df, ELEMENTS_TO_KEEP_NO_FE)

# ### Train test split, datasets, dataloaders

X_y_train, X_y_val, X_y_test = create_partition(inDKs_df)


# ### Creating features and labels matrices


X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = prepare_data_for_training(
                                                                                                                X_y_train, 
                                                                                                                X_y_val, 
                                                                                                                X_y_test, 
                                                                                                                ELEMENTS_TO_KEEP_NO_FE)


# ### Normalization / taking logarithm

X_train = transform_data(X_train, PREPROCESSING_METHOD)
y_train = transform_data(y_train, PREPROCESSING_METHOD)
X_val = transform_data(X_val, PREPROCESSING_METHOD)
y_val = transform_data(y_val, PREPROCESSING_METHOD)
X_test = transform_data(X_test, PREPROCESSING_METHOD)
y_test = transform_data(y_test, PREPROCESSING_METHOD)


dim = 8
plt.hist(X_train[:,dim], bins=100);
plt.hist(X_val[:,dim], bins=100);
plt.hist(X_test[:,dim], bins=100);


# ### Converting to tensors

device = get_device()

X_train = data_to_device(X_train, device)
y_train = data_to_device(y_train, device)
X_val = data_to_device(X_val, device)
y_val = data_to_device(y_val, device)
X_test = data_to_device(X_test, device)
y_test = data_to_device(y_test, device)


# ### Dataset

train_dataset = InksDataset(X=X_train, y=y_train)
val_dataset = InksDataset(X=X_val, y=y_val)
test_dataset = InksDataset(X=X_test, y=y_test)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=100)
val_loader = DataLoader(val_dataset, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=False)


# ## Neural network

model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)


# Cu >> Mn > Al > Zn > Pb > S > Cr > Co >>


weights = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(device)
loss_fn = CustomLoss(weights=weights)


# model, train_losses, val_losses = train_model(model=model,
#                                               train_loader=train_loader,
#                                               val_loader=val_loader,
#                                               epochs=5000,
#                                               loss_fn=loss_fn)


# ### Loss visualisation


# visualise_losses(train_losses, val_losses)


# ## Loading pretrained model

model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
model.load_state_dict(torch.load(MODELS_PATH))


# ## Prediction on test set

labels, outputs, difference, mean_loss = evaluate_on_test_set(model=model, 
                                                              test_loader=test_loader, 
                                                              loss_fn=loss_fn)


# Consecutive inks from test set are in labels tensor (498 examples, 10 elements/coordinates):

print(labels)


# Predictions are in out tensor (498 examples, 10 elements/coordinates):

print(outputs)


# Absolute value of difference between true values (labels) and prediction (outputs) are stored in difference tensor (498 examples, 10 elements/coordinates):

print(difference)


# Mean loss:

print(mean_loss)


print(torch.mean(difference, axis=0))


# Consecutive coordinates correspond to: Al, S, Cr, Mn, Co, Cu, Zn, Pb, Fe, Mg.
# 
# Significance of elements: Cu >> Mn > Al > Zn > Pb > S > Cr > Co >> all the others.

# ## Saving the model

# save_model(model, MODELS_PATH, HOW_MANY_OUTER_TO_REMOVE, 
#                 ELEMENTS_TO_KEEP, PREPROCESSING_METHOD)


# ### Visualisations

visualise_prediction_against_true(outputs, labels, dimension=0)


# ## Quality of prediction: closest points

# ### Creating df with means of classes

mean_df = create_means_df(ELEMENTS_TO_KEEP_NO_FE, y_train.cpu(), train_order, y_val.cpu(), val_order)


# ### Checking if there is any chance for it to work: if classes in y_test are close to means of classes in train_val set


y_test_df, consistency = find_and_compare_closest_class(y_test.cpu(), test_order, ELEMENTS_TO_KEEP_NO_FE, mean_df)


print(consistency)


# ### Now let's check if it works for the neural network's output.


model.eval()
outputs = model(X_test)

outputs_df, accuracy = find_and_compare_closest_class(outputs.cpu().detach().numpy(), test_order, ELEMENTS_TO_KEEP_NO_FE, mean_df)


accuracy = compute_accuracy(outputs_df)


precision, recall = compute_precision_and_recall(outputs_df)



plot_confusion_matrix(true_labels = outputs_df['Real sample_id'], 
                      closest_labels = outputs_df['Closest sample id'], 
                      normalization_in_conf_mat = None, 
                      path_to_save = FIGURES_PATH)


# ## Quality of prediction: confidence intervals

model.eval()
outputs = model(X_test)

min_df, max_df, min_max_df = create_min_max_df(y_train.cpu(), 
                                               train_order, 
                                               y_val.cpu(), 
                                               val_order, 
                                               ELEMENTS_TO_KEEP_NO_FE)


lower_bound_df, upper_bound_df = create_mean_plus_sd_df(y_train.cpu(), 
                                                        train_order, 
                                                        y_val.cpu(), 
                                                        val_order, 
                                                        ELEMENTS_TO_KEEP_NO_FE, 
                                                        how_many_sd=2)


min_max_res_list, fraction_1 = is_inside_interval(outputs, test_order, min_df, max_df)


sd_res_list, fraction_2 = is_inside_interval(outputs, test_order, lower_bound_df, upper_bound_df)


# Percentage of points inside min-max interval:

print(fraction_1)


# On consecutive coordinates:

np.array(min_max_res_list).mean(0)


# Percentage of points inside mean +- 2 standard deviations interval:

print(fraction_2)


# On consecutive coordinates:

np.array(sd_res_list).mean(0)


# ### Visualisations of min-max intervals

visualise_min_max_intervals(outputs, 
                            test_order, 
                            ELEMENTS_TO_KEEP_NO_FE, 
                            min_max_df, 
                            min_max_res_list, 
                            element_nr=2, 
                            path_to_save=None)


# ### Visualisations of mean +- 2 * standard deviation intervals


visualise_mean_plus_minus_sd_intervals(outputs, 
                                       test_order, 
                                        ELEMENTS_TO_KEEP_NO_FE, 
                                       lower_bound_df, 
                                       upper_bound_df, 
                                       how_many_sd=2,
                                       sd_res_list=sd_res_list, 
                                       element_nr=2, 
                                       path_to_save=None)

