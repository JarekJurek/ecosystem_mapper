import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class VariablesModel(nn.Module):
    """
    Model to classify based on the variables
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, use_batchnorm):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)


    def forward(
        self, images: Optional[torch.Tensor], variables: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if variables is None:
            raise ValueError("VariablesOnlyModel requires 'variables' input tensor.")

        x = self.mlp(variables)
        logits = self.fc_out(x)
        return logits
