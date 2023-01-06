from abc import abstractmethod

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from src import config, Logger


class GANBase:

    def __init__(
            self,
            generator: Module,
            discriminator: Module,
            generator_optimizer: Optimizer,
            discriminator_optimizer: Optimizer,
            training_config,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.training_config = training_config
        self.statistics = {
            'discriminator_loss': [],
            'generator_loss': [],
        }

    def train(self, dataset: Dataset):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')

        x = dataset[:][0].to(config.device)
        for _ in tqdm(range(self.training_config.epochs)):
            loss = 0
            for _ in range(self.training_config.discriminator_loop_num):
                loss = self._train_discriminator(x)
            self.statistics['discriminator_loss'].append(loss)
            for _ in range(self.training_config.generator_loop_num):
                loss = self._train_generator(len(x))
            self.statistics['generator_loss'].append(loss)
        self.generator.eval()
        self.discriminator.eval()
        self._save_model()
        self._plot()
        self.logger.info(f'Finished training')

    @abstractmethod
    def _train_discriminator(self, x: torch.Tensor) -> float:
        pass

    @abstractmethod
    def _train_generator(self, x_len: int) -> float:
        pass

    def _save_model(self):
        generator_path = config.path.data / f'{self.__class__.__name__}_generator.pt'
        torch.save(self.generator.state_dict(), generator_path)
        self.logger.debug(f'Saved generator model at {generator_path}')

        discriminator_path = config.path.data / f'{self.__class__.__name__}_discriminator.pt'
        torch.save(self.discriminator.state_dict(), discriminator_path)
        self.logger.debug(f'Saved discriminator model at {discriminator_path}')

    def _plot(self):
        sns.set()
        plt.title(f"{self.__class__.__name__} Generator and Discriminator Loss During Training")
        plt.plot(self.statistics['generator_loss'], label="Generator")
        plt.plot(self.statistics['discriminator_loss'], label="Discriminator")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = config.path.plots / f'{self.__class__.__name__}_Loss.png'
        plt.savefig(fname=str(plot_path))
        plt.clf()
        self.logger.debug(f'Saved plot at {plot_path}')

    def load_model(self):
        generator_path = config.path.data / f'{self.__class__.__name__}_generator.pt'
        self.generator.load_state_dict(
            torch.load(generator_path)
        )
        self.generator.to(config.device)
        self.generator.eval()
        self.logger.debug(f'Loaded generator model from {generator_path}')

        discriminator_path = config.path.data / f'{self.__class__.__name__}_discriminator.pt'
        self.discriminator.load_state_dict(
            torch.load(discriminator_path)
        )
        self.discriminator.to(config.device)
        self.discriminator.eval()
        self.logger.debug(f'Loaded discriminator model from {discriminator_path}')
