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
NORMALISATION_TO_FE = config["normalisation_to_Fe"]
MULTIPLICATION_WEIGHTS = config["multiplication_weights"]
DROPOUT_PROB = config["dropout_prob"]

DATA_PATH = config["training_data_path"]
MODELS_PATH = config["models_path"]
FIGURES_PATH = config["figures_path"]["training"]

## Loading data and trained model

# Preparing training, validation, test data

train_loader, val_loader, test_loader, data_list = prepare_training_data(
                                                                        data_path = DATA_PATH,
                                                                        how_many_outer_to_remove = HOW_MANY_OUTER_TO_REMOVE,
                                                                        elements_to_keep = ELEMENTS_TO_KEEP,
                                                                        multiplication_weights = MULTIPLICATION_WEIGHTS,
                                                                        preprocessing_method = PREPROCESSING_METHOD, 
                                                                        return_data=True,
                                                                        normalisation_to_Fe=NORMALISATION_TO_FE
                                                                        )

X_train, y_train, X_val, y_val, X_test, y_test, train_order, val_order, test_order = data_list

device = get_device()
X_train = data_to_device(X_train, device)
y_train = data_to_device(y_train, device)
X_val = data_to_device(X_val, device)
y_val = data_to_device(y_val, device)
X_test = data_to_device(X_test, device)
y_test = data_to_device(y_test, device)

# Loading pretrained model

model = InksNet(input_size=INPUT_SIZE, dropout_prob=DROPOUT_PROB).to(device)
model.load_state_dict(torch.load(MODELS_PATH, weights_only=False))

# Defining loss function

weights = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(device)
loss_fn = CustomLoss(weights=weights)

# Prediction on test set

labels, outputs, difference, mean_loss = evaluate_on_test_set(model=model, 
                                                              test_loader=test_loader, 
                                                              loss_fn=loss_fn)


## Validation on test set (never seen by InksNet)

# Consecutive inks from test set are in labels tensor:
print('y_true:')
print(labels)


# Predictions are in outputs tensor:
print('y_pred:')
print(outputs)


# Absolute value of difference between true values (labels) and prediction (outputs) are stored in difference tensor:
print('Difference:')
print(difference)

print('Mean of |y_pred/y_true|:')
for i, element in enumerate(ELEMENTS_TO_KEEP):
    plt.plot(np.exp(to_numpy(outputs)[:,i])/np.exp(to_numpy(labels)[:,i]))
    plt.title(element)
    #plt.ylim((0.8, 1.2))
    plt.axhline(0.8)
    plt.axhline(1.2)
    plt.show()

    print(element)
    print('Fraction above 120%:')
    print(((np.exp(to_numpy(outputs)[:,i])/np.exp(to_numpy(labels)[:,i]))>1.2).sum() / outputs.shape[0])
    print('Fraction below 80%:')
    print(((np.exp(to_numpy(outputs)[:,i])/np.exp(to_numpy(labels)[:,i]))<0.8).sum() / outputs.shape[0])
    print('Mean deviation from 1:')
    print(np.abs(1-(np.exp(to_numpy(outputs)[:,i])/np.exp(to_numpy(labels)[:,i]))).mean())
    print('--------------------')


# for i, element in enumerate(ELEMENTS_TO_KEEP):
#     plt.plot((to_numpy(outputs)-to_numpy(labels))[:,i])
#     plt.title(element)
#     plt.show()

# Mean loss:
print('Mean loss:')
print(mean_loss)

# Loss on consecutive coordinates:
print('Loss on consecutive coordinates:')
print(torch.mean(difference, axis=0))
# Consecutive coordinates correspond to: Al, S, Cr, Mn, Co, Cu, Zn, Pb, Fe, Mg.
# Significance of elements: Cu >> Mn > Al > Zn > Pb > S > Cr > Co >> all the others.

# Visualise prediction against true:

element_nr = 5 #Cu
visualise_prediction_against_true(outputs, labels, dimension=element_nr)


# Quality of prediction: closest points

# Creating df with means of classes

ELEMENTS_TO_KEEP_NO_FE = [el for el in ELEMENTS_TO_KEEP if el != 'Fe']
mean_df = create_means_df(ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, y_train.cpu(), train_order, y_val.cpu(), val_order)

# Checking if there is any chance for it to work: if classes in y_test are close to means of classes in train_val set

y_test_df, consistency = find_and_compare_closest_class(y_test.cpu(), test_order, ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, mean_df)
print('For y_true, the fraction of samples for which the closest class center is the center of true class (a.k.a. accuracy) is equal to:')
print(consistency)


# Checking how it works for the neural network's output.

model.eval()
outputs = model(X_test)
outputs_df, accuracy = find_and_compare_closest_class(outputs.cpu().detach().numpy(), test_order, ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, mean_df)

print('Whereas for y_pred:')
accuracy = compute_accuracy(outputs_df)

# Checking precision and recall

print('How about precision and recall?')
precision, recall = compute_precision_and_recall(outputs_df)


plot_confusion_matrix(true_labels = outputs_df['Real sample_id'], 
                      closest_labels = outputs_df['Closest sample id'], 
                      normalization_in_conf_mat = None, 
                      path_to_save = FIGURES_PATH)


# Quality of prediction: confidence intervals

how_many_sd = 2

model.eval()
outputs = model(X_test)

min_df, max_df, min_max_df = create_min_max_df(y_train.cpu(), 
                                               train_order, 
                                               y_val.cpu(), 
                                               val_order, 
                                               ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP)


lower_bound_df, upper_bound_df = create_mean_plus_sd_df(y_train.cpu(), 
                                                        train_order, 
                                                        y_val.cpu(), 
                                                        val_order, 
                                                        ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, 
                                                        how_many_sd=how_many_sd)


min_max_res_list, fraction_1 = is_inside_interval(outputs, test_order, min_df, max_df)


sd_res_list, fraction_2 = is_inside_interval(outputs, test_order, lower_bound_df, upper_bound_df)


# Fraction of points inside min-max interval:

print('Fraction of points inside min-max interval:')
print(fraction_1)


# On consecutive coordinates:

np.array(min_max_res_list).mean(0)


# Fraction of points inside mean +- how_many_sd standard deviations interval:

print('Fraction of points inside mean +-' + str(how_many_sd) + 'standard deviations interval:')
print(fraction_2)


# On consecutive coordinates:

np.array(sd_res_list).mean(0)


# Visualisations of min-max intervals

element_nr = 0 #Al

# visualise_min_max_intervals(outputs, 
#                             test_order, 
#                             ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, 
#                             min_max_df, 
#                             min_max_res_list, 
#                             element_nr=element_nr, 
#                             path_to_save=None)


# Visualisations of mean +- 2 * standard deviation intervals


# visualise_mean_plus_minus_sd_intervals(outputs, 
#                                        test_order, 
#                                         ELEMENTS_TO_KEEP_NO_FE if NORMALISATION_TO_FE else ELEMENTS_TO_KEEP, 
#                                        lower_bound_df, 
#                                        upper_bound_df, 
#                                        how_many_sd=2,
#                                        sd_res_list=sd_res_list, 
#                                        element_nr=element_nr, 
#                                        path_to_save=None)

