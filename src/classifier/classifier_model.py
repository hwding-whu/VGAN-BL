import torch
from torch import nn

from src.config import data
from src.utils import init_weights


class ClassifierModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(data.x_size, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Linear(8, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Linear(4, 2, bias=False),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        prob = self.process(x)
        return prob
