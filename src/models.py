import torch.nn as nn
from math import sqrt
import torch

class MLP(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_dim=32):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        return self.net(x)