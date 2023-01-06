import numpy.random

import context

import random

import numpy as np
import torch

from src import config, utils


if __name__ == '__main__':
    utils.set_random_state()
    old_ele = random.random()
    new_ele = random.random()
    assert old_ele != new_ele

    utils.set_random_state()
    new_ele = random.random()
    assert old_ele == new_ele

    config.seed = 1
    utils.set_random_state()
    new_ele = random.random()
    assert old_ele != new_ele

