import torch
from torch import nn
from torch.utils.data import Dataset
from data_utils import get_device

DROPOUT_PROB = 0.1
INPUT_SIZE = 10
WEIGHTS = torch.tensor([1/INPUT_SIZE]*INPUT_SIZE).unsqueeze(1).to(get_device())

class InksNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.Dropout(DROPOUT_PROB),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.Dropout(DROPOUT_PROB),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 1024),
        nn.Dropout(DROPOUT_PROB),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.Dropout(DROPOUT_PROB),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.Dropout(DROPOUT_PROB),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, INPUT_SIZE))
    def forward(self, x):
        return self.seq(x)
    

class CustomLoss(nn.Module):
    def __init__(self, weights=WEIGHTS):
        super(CustomLoss, self).__init__()
        self.weights = weights
        assert weights.shape[1] == 1

    def forward(self, outputs, targets):
        inner_loss = nn.L1Loss(reduction='none')
        res = inner_loss(outputs, targets)
        res = torch.mm(res, self.weights)
        res = res.mean(axis=0)
#         inner_loss_to_compare = nn.L1Loss(reduction='mean')
#         assert torch.isclose(inner_loss_to_compare(outputs, targets), res)
        return res