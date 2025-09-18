import torch
from torch import nn
from data_utils import get_device, CustomLoss

INPUT_SIZE = 10
WEIGHTS = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(get_device())
EPOCHS = 5000


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


def train_model(model, train_loader, val_loader, loss_fn=CustomLoss()):

    #loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_number = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(train_loader, loss_fn=loss_fn, optimizer=optimizer)


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