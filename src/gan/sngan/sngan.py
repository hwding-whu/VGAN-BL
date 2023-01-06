import torch

from src import config
from src.gan._gan_base import GANBase
from .models import DiscriminatorModel, GeneratorModel


class SNGAN(GANBase):

    def __init__(self):
        generator = GeneratorModel().to(config.device)
        discriminator = DiscriminatorModel().to(config.device)
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=torch.optim.Adam(
                params=generator.parameters(),
                lr=config.training.sngan.generator_lr,
                betas=(0.5, 0.9),
            ),
            discriminator_optimizer=torch.optim.Adam(
                params=discriminator.parameters(),
                lr=config.training.sngan.discriminator_lr,
                betas=(0.5, 0.9),
            ),
            training_config=config.training.sngan,
        )

    def _train_discriminator(self, x: torch.Tensor) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _train_generator(self, x_len: int) -> float:
        self.generator.zero_grad()
        z = torch.randn(x_len, config.data.z_size, device=config.device)
        fake_x = self.generator(z)
        prediction = self.discriminator(fake_x)
        loss = - prediction.mean()
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()
