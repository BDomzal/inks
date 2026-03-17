import torch
from torch import nn
import numpy as np

class InksNet(nn.Module):
    """Small fully-connected network used for reconstruction/regression.

    The architecture is a stack of Linear -> Dropout -> BatchNorm1d -> ReLU
    blocks with expanding and contracting widths and a final Linear layer that
    projects back to ``input_size``. The module does not apply any final
    activation so it can be used for regression.

    Parameters
    ----------
    input_size:
        Number of features in the input (and output) tensor.
    dropout_prob:
        Dropout probability used after each Linear layer.
    """

    # def __init__(self, input_size: int, dropout_prob: float) -> None:
    #     super().__init__()
    #     self.seq = nn.Sequential(
    #         nn.Linear(input_size, 64),
    #         nn.Dropout(dropout_prob),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.Linear(64, 256),
    #         nn.Dropout(dropout_prob),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Linear(256, 1024),
    #         nn.Dropout(dropout_prob),
    #         nn.BatchNorm1d(1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, 256),
    #         nn.Dropout(dropout_prob),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Linear(256, 64),
    #         nn.Dropout(dropout_prob),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.Linear(64, input_size),
    #     )

    def __init__(self, input_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 765),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(765, 402),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(402, 311),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(311, 547),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(547, 365),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(365, 78),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(78, 341),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(341, input_size),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input.
        """
        return self.seq(x)


class CustomLoss(nn.Module):
    """Weighted L1 loss that reduces per-feature L1 to a single value.

    The loss computes elementwise L1 distances between ``outputs`` and
    ``targets`` then performs a matrix multiplication with the provided
    ``weights`` vector to obtain a per-sample scalar which is finally averaged
    across the batch.

    Parameters
    ----------
    weights:
        1D tensor of per-feature weights (shape (features,)) or a 2D tensor
        of shape (features, 1). Internally this is stored as shape (features, 1).
    """

    def __init__(self, weights: torch.Tensor) -> None:
        super(CustomLoss, self).__init__()
        if weights.dim() == 1:
            assert weights.shape[0] > 0, "Weights tensor must not be empty"
            self.weights = weights.unsqueeze(1)
        elif weights.dim() == 2 and weights.shape[1] == 1:
            self.weights = weights
        else:
            raise ValueError("weights must be a 1D tensor or a 2D tensor with shape (N, 1)")
        #self.inner_loss = nn.L1Loss(reduction='none')
        #self.inner_loss = nn.MSELoss(reduction='none')
        self.inner_loss = nn.HuberLoss(reduction='none', delta=1)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted L1 loss.

        Parameters
        ----------
        outputs, targets:
            Tensors of shape (batch_size, features).

        Returns
        -------
        torch.Tensor
            Reduced tensor (scalar-like) containing the mean weighted L1 loss.
        """
        
        res = self.inner_loss(outputs, targets)
        #IMPORTANT CHANGE BELOW!!! UN-LOGARITHMING THE VALUES
        #res = self.inner_loss(torch.exp(outputs), torch.exp(targets))

        res = torch.mm(res, self.weights)
        res = res.mean(axis=0)
#         inner_loss_to_compare = nn.L1Loss(reduction='mean')
#         assert torch.isclose(inner_loss_to_compare(outputs, targets), res)
        return res

class CustomLoss2(nn.Module):


    def __init__(self) -> None:
        super(CustomLoss2, self).__init__()

        #self.inner_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, acceptable_interval=(np.log(0.8), np.log(1.2)), gentleness=0.01) -> torch.Tensor:

        ratio = outputs - targets #logarithmic scale so subtraction instead of division

        #is_inside_interval = 0.5*(torch.tanh(1+(1/gentleness)*(ratio-acceptable_interval[0]))-torch.tanh(1+(1/gentleness)*(ratio-acceptable_interval[1]))) #continuous approximation of indicator

        is_inside_interval = torch.exp((-ratio**2)/0.05)


        is_outside_interval = 1 - is_inside_interval
        return is_outside_interval.mean()

class CustomLoss3(nn.Module):


    def __init__(self) -> None:
        super(CustomLoss3, self).__init__()

        #self.inner_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, gentleness=0.01) -> torch.Tensor:

        ratio = outputs - targets #logarithmic scale so subtraction instead of division
        rel_error = torch.abs(ratio)
        penalty = torch.relu(rel_error - np.min([np.abs(np.log(0.8)), np.abs(np.log(1.2))]))

        return torch.mean(penalty ** 2)


class CustomLoss4(nn.Module):


    def __init__(self) -> None:
        super(CustomLoss4, self).__init__()

        #self.inner_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        penalty = nn.functional.huber_loss(torch.exp(outputs)/torch.abs(torch.exp(targets)), torch.exp(targets)/torch.abs(torch.exp(targets)), reduction='mean', delta=0.5)

        return penalty

class CustomLoss5(nn.Module):


    def __init__(self) -> None:
        super(CustomLoss5, self).__init__()

        #self.inner_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        penalty = nn.functional.mse_loss(torch.exp(outputs)/torch.abs(torch.exp(targets)), torch.exp(targets)/torch.abs(torch.exp(targets)), reduction='mean')

        return penalty