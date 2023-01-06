import context

import torch

from src import utils, config
from src.dataset import MinorityDataset
from src import WGANGP


FILE_NAME = 'page-blocks0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)
    # set config
    utils.set_x_size()
    # train
    dataset = MinorityDataset(training=True)
    utils.set_random_state()
    wgangp = WGANGP()
    wgangp.train(dataset=dataset)
    # test
    wgangp.load_model()
    z = torch.randn(1, config.data.z_size, device=config.device)
    print(wgangp.generator(z))
