"""
model.py
========

Neural network model used in FLECHA.

The model is a multilayer perceptron for connectome-based regression. It
preserves the architecture used in the original experiment script.
"""

import torch
from torch import nn
import torch.nn.init as init


class ConnectomeRegressor4L(nn.Module):
    """
    Four-layer connectome regressor.

    Parameters
    ----------
    input_size : int
        Number of connectome edge features.
    n_l1, n_l2, n_l3 : int
        Hidden layer widths.
    dropout : float
        Dropout probability.
    """

    def __init__(self, input_size, n_l1=256, n_l2=128, n_l3=64, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),

            nn.Dropout(dropout),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),

            nn.Dropout(dropout),
            nn.Linear(n_l2, n_l3),
            nn.ReLU(),
            nn.BatchNorm1d(n_l3),

            nn.Dropout(dropout),
            nn.Linear(n_l3, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following the original implementation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        return self.net(x)
