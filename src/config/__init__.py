import torch

import random

from . import (
    training,
    data,
    dataset,
    path,
)

# random seed
seed: int = random.randint(1, 10000)

# pytorch device
device = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
