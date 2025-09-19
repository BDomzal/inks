import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import CustomLoss, get_device
from model import InksNet
import time
import datetime

def train_one_epoch(model, loader, loss_fn, optimizer):
    running_loss = 0.

    for i, batch in enumerate(loader):

        inputs, labels = batch ###

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader) # loss per batch
    return avg_loss


def train_model(model, train_loader, val_loader, epochs, loss_fn=CustomLoss()):

    #loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_number = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(model, train_loader, loss_fn=loss_fn, optimizer=optimizer)


        running_vloss = 0.0
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                inputs, labels = vdata
                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
                running_vloss += loss.item()
        
        avg_vloss = float(running_vloss / len(val_loader))
        print('LOSS train {} test {}'.format(avg_loss, avg_vloss))
        
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss)

        epoch_number += 1

    return model, train_losses, val_losses


def visualise_losses(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)


def load_saved_model(models_path):
    model = InksNet().to(get_device())
    model.load_state_dict(torch.load(models_path))
    return model


def evaluate_on_test_set(model, test_loader, loss_fn=CustomLoss()):
    
    model.eval()
    running_tloss = 0

    for i, tdata in enumerate(test_loader):
        inputs, labels = tdata
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if i==0:
            out = outputs
            lab = labels
            res = abs(outputs-labels)
        else:
            res = torch.concat((res, abs(outputs-labels)), axis=0)
            lab = torch.concat((lab, labels), axis=0)
            out = torch.concat((out, outputs), axis=0)
        running_tloss += loss.item()
        
    mean_loss = running_tloss/len(test_loader)
    return lab, out, res, mean_loss


def save_model(model, models_path, remove_outer, how_many_outer_to_remove, 
                elements_to_keep, preprocessing_method):

    outer_removed = 'outer_removed' if remove_outer else 'outer_not_removed'

    elements = '_'.join(elements_to_keep)
    settings_str = '_' + preprocessing_method + '_' + \
                    outer_removed + '_' + str(how_many_outer_to_remove) + \
                    '_' + elements + '_'

    ts = time.time()
    now = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

    torch.save(model.state_dict(), '/'.join(models_path.split('/')[:-1]) + '/model_regression' + settings_str + now)

def visualise_prediction_against_true(outputs, labels, dimension):

    selected_outputs = outputs[:, dimension].cpu().detach().numpy()
    selected_labels = labels[:, dimension].cpu().detach().numpy()
    plt.plot(selected_labels, selected_outputs, '.')
    plt.plot( [-10, 10],[-10, 10], 'r' )
    plt.xlabel('True value')
    plt.ylabel('Predicted value')

    
# SPRAWDZIĆ TO PONIŻEJ
def create_means_df(elements_to_keep, y_train, train_order, y_val, val_order):

    columns_to_keep_inds = [el + '_i' for el in elements_to_keep]

    train_val_df = pd.DataFrame(y_train.cpu())
    train_val_df = pd.concat([train_val_df, 
                                        pd.DataFrame(y_val.cpu())])
    train_val_df.columns = elements_to_keep
    train_val_df.insert(0, 'Sample_id', pd.concat([train_order, val_order]))
    train_val_df.reset_index(drop=True, inplace=True)

    mean_df = train_val_df[['Sample_id']]

    for name in elements_to_keep:
        mean_df[name] = train_val_df.groupby('Sample_id')[name].transform('mean')
    mean_df.drop_duplicates('Sample_id', inplace=True)

    mean_df['Sample_id'] = mean_df['Sample_id'].astype(str)
    mean_df.sort_values(by='Sample_id', inplace=True)
    mean_df.reset_index(drop=True, inplace=True)
    return mean_df
    
