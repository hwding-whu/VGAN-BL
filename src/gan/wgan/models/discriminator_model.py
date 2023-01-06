import torch
from torch import nn

from src import config
from src.utils import init_weights


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(config.data.x_size, 512, bias=False),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32, bias=False),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8, bias=False),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat

