import torch
from torch import nn

class InksNet(nn.Module):
    def __init__(self, input_size, dropout_prob):
        super().__init__()
        self.seq = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.Dropout(dropout_prob),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.Dropout(dropout_prob),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 1024),
        nn.Dropout(dropout_prob),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.Dropout(dropout_prob),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.Dropout(dropout_prob),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, input_size))
    def forward(self, x):
        return self.seq(x)
    

class CustomLoss(nn.Module):
    
    def __init__(self, weights):
        super(CustomLoss, self).__init__()
        if weights.dim() == 1:
            assert weights.shape[0] > 0, "Weights tensor must not be empty"
            self.weights = weights.unsqueeze(1)
        elif weights.dim() == 2 and weights.shape[1] == 1:
            self.weights = weights
        else:
            raise ValueError("weights must be a 1D tensor or a 2D tensor with shape (N, 1)")
        self.inner_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs, targets):
        res = self.inner_loss(outputs, targets)
        res = torch.mm(res, self.weights)
        res = res.mean(axis=0)
#         inner_loss_to_compare = nn.L1Loss(reduction='mean')
#         assert torch.isclose(inner_loss_to_compare(outputs, targets), res)
        return res