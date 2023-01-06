import context

import torch

from src import utils, config
from src.dataset import MinorityDataset
from src import GAN


FILE_NAME = 'segment0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)
    # set config
    utils.set_x_size()
    # train
    dataset = MinorityDataset(training=True)
    utils.set_random_state()
    gan = GAN()
    gan.train(dataset=dataset)
    # test
    gan.load_model()
    z = torch.randn(1, config.data.z_size, device=config.device)
    print(gan.generator(z))
