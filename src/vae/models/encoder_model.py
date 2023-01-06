import torch
from torch import nn

from src.config import data
from src.utils import init_weights


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calculate_mu = nn.Sequential(
            nn.Linear(data.x_size, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, data.z_size),
        )
        self.calculate_log_variance = nn.Sequential(
            nn.Linear(data.x_size, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, data.z_size),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        mu = self.calculate_mu(x)
        log_variance = self.calculate_log_variance(x)
        sigma = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(mu)
        z = epsilon * sigma + mu
        return z, mu, sigma
