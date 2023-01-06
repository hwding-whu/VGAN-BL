import torch
from torch import nn

from src import config
from src.utils import init_weights


class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(config.data.z_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, config.data.x_size),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat

