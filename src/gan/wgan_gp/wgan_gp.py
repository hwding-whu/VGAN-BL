import torch

from src import config
from src.gan._gan_base import GANBase
from .models import DiscriminatorModel, GeneratorModel


class WGANGP(GANBase):

    def __init__(self):
        generator = GeneratorModel().to(config.device)
        discriminator = DiscriminatorModel().to(config.device)
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=torch.optim.Adam(
                params=generator.parameters(),
                lr=config.training.wgan_gp.generator_lr,
                betas=(0.5, 0.9),
            ),
            discriminator_optimizer=torch.optim.Adam(
                params=discriminator.parameters(),
                lr=config.training.wgan_gp.discriminator_lr,
                betas=(0.5, 0.9),
            ),
            training_config=config.training.wgan_gp,
        )

    def _train_discriminator(self, x: torch.Tensor) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        gradient_penalty = self._cal_gradient_penalty(x, fake_x)
        loss = loss_real + loss_fake + gradient_penalty
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

    def _cal_gradient_penalty(
            self,
            x: torch.Tensor,
            fake_x: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.rand(len(x), 1).to(config.device)
        interpolates = alpha * x + (1 - alpha) * fake_x
        interpolates.requires_grad = True
        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(config.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.training.wgan_gp.gp_lambda
        return gradient_penalty
