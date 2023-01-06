import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from tqdm import tqdm

from src import config, Logger
from src.vae.models import EncoderModel, DecoderModel


class VAE:

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.encoder = EncoderModel().to(config.device)
        self.decoder = DecoderModel().to(config.device)
        self.statistics = {
            'Loss': []
        }

    def train(self, dataset):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=config.training.vae.encoder_lr,
        )
        decoder_optimizer = torch.optim.Adam(
            params=self.decoder.parameters(),
            lr=config.training.vae.decoder_lr,
        )
        x = dataset[:][0]
        x = x.to(config.device)
        for _ in tqdm(range(config.training.vae.epochs)):
            # clear gradients
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            # calculate z, mu and sigma
            z, mu, sigma = self.encoder(x)
            # calculate x_hat
            x_hat = self.decoder(z)
            # calculate loss
            divergence = - 0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
            loss = divergence + mse_loss(x_hat, x)
            # calculate gradients
            loss.backward()
            self.statistics['Loss'].append(loss.item())
            # optimize models
            encoder_optimizer.step()
            decoder_optimizer.step()

        self.encoder.eval()
        self.decoder.eval()
        self._plot()
        self._save_model()
        self.logger.info("Finished training")

    def _plot(self):
        sns.set()
        plt.title(f"{self.__class__.__name__} Loss During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        sns.lineplot(data=self.statistics['Loss'])
        plot_path = config.path.plots / f'{self.__class__.__name__}_Loss.png'
        plt.savefig(plot_path)
        plt.clf()
        self.logger.debug(f'Saved plot at {plot_path}')

    def _save_model(self):
        encoder_path = config.path.data / f'{self.__class__.__name__}_encoder.pt'
        torch.save(self.encoder.state_dict(), encoder_path)
        self.logger.debug(f'Saved encoder model at {encoder_path}')

        decoder_path = config.path.data / f'{self.__class__.__name__}_decoder.pt'
        torch.save(self.decoder.state_dict(), decoder_path)
        self.logger.debug(f'Saved decoder model at {decoder_path}')

    def load_model(self):
        encoder_path = config.path.data / f'{self.__class__.__name__}_encoder.pt'
        self.encoder.load_state_dict(
            torch.load(encoder_path)
        )
        self.encoder.to(config.device)
        self.encoder.eval()
        self.logger.debug(f'Loaded encoder model from {encoder_path}')

        decoder_path = config.path.data / f'{self.__class__.__name__}_decoder.pt'
        self.decoder.load_state_dict(
            torch.load(decoder_path)
        )
        self.decoder.to(config.device)
        self.decoder.eval()
        self.logger.debug(f'Loaded decoder model from {decoder_path}')
