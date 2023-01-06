import context

import torch

from src import utils
from src.dataset import CompleteDataset, MinorityDataset
from src import Classifier, SNGAN, VAE


FILE_NAME = 'page-blocks0.dat'

if __name__ == '__main__':

    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # set config
    utils.set_x_size()

    # normally train
    utils.set_random_state()
    classifier = Classifier('Test_Normal_Train')
    classifier.train(
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )
    for name, value in utils.get_final_test_metrics(classifier.statistics).items():
        print(f'{name:<10}:{value:>10}')

    # train with generator
    utils.set_random_state()
    sn_gan = SNGAN()
    sn_gan.train(MinorityDataset(training=True))
    sn_gan.load_model()

    utils.set_random_state()
    classifier = Classifier('Test_G_Train')
    classifier.g_train(
        generator=sn_gan.generator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )
    for name, value in utils.get_final_test_metrics(classifier.statistics).items():
        print(f'{name:<10}:{value:>10}')

    # train with encoder, generator and discriminator
    utils.set_random_state()
    vae = VAE()
    vae.train(MinorityDataset(training=True))
    vae.load_model()
    utils.set_random_state()
    classifier = Classifier('Test_EGD_Train')
    classifier.egd_train(
        encoder=vae.encoder,
        generator=sn_gan.generator,
        discriminator=sn_gan.discriminator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
        seed_dataset=MinorityDataset(training=True),
    )
    for name, value in utils.get_final_test_metrics(classifier.statistics).items():
        print(f'{name:<10}:{value:>10}')
