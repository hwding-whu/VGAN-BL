import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src import config
from src.utils import init_weights


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(config.data.x_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.process(x)
