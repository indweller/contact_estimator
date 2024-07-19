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
    

class MLP3(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_dim=[32, 32, 32]):
        super(MLP3, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)