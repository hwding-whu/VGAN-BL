import torch

from src import config, utils
from src.gan._gan_base import GANBase
from .models import DiscriminatorModel, GeneratorModel


class GANHL(GANBase):

    def __init__(self):
        generator = GeneratorModel().to(config.device)
        discriminator = DiscriminatorModel().to(config.device)
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=torch.optim.Adam(
                params=generator.parameters(),
                lr=config.training.gan_hl.generator_lr,
                betas=(0.5, 0.999),
            ),
            discriminator_optimizer=torch.optim.Adam(
                params=discriminator.parameters(),
                lr=config.training.gan_hl.discriminator_lr,
                betas=(0.5, 0.999),
            ),
            training_config=config.training.gan_hl,
        )
        self.statistics['hidden_loss'] = []

    def _train_discriminator(self, x: torch.Tensor) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = -torch.log(prediction_real.mean())
        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = -torch.log(1 - prediction_fake.mean())
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _train_generator(self, x_len: int) -> float:
        self.generator.zero_grad()

        # get the hidden output of real x
        real_x_hidden_output = self.discriminator.hidden_output.detach()

        # get the final output and hidden output of fake x
        z = torch.randn(x_len, config.data.z_size, device=config.device)
        fake_x = self.generator(z)
        final_output = self.discriminator(fake_x)
        fake_x_hidden_output = self.discriminator.hidden_output

        cal_kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        real_x_hidden_distribution = utils.normalize(real_x_hidden_output)
        fake_x_hidden_distribution = utils.normalize(fake_x_hidden_output)
        hidden_loss = cal_kl_div(
            input=fake_x_hidden_distribution,
            target=real_x_hidden_distribution,
        ) * config.training.gan_hl.hl_lambda

        self.statistics['hidden_loss'].append(hidden_loss.item())
        loss = -torch.log(final_output.mean()) + hidden_loss
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()
