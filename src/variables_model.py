import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class VariablesOnlyModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        dropout,
        use_batchnorm,
    ):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        elif isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)

        layers = []
        prev_dim = input_dim

        # build all hidden layers dynamically
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = h  # next layer input

        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, images: Optional[torch.Tensor], variables: Optional[torch.Tensor]):
        if variables is None:
            raise ValueError("VariablesOnlyModel requires 'variables' input tensor.")

        x = self.mlp(variables)
        logits = self.fc_out(x)
        return logits
