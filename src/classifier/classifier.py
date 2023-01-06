import random
from math import sqrt

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src import config, Logger
from src.classifier.classifier_model import ClassifierModel


class Classifier:

    def __init__(
            self,
            name: str,
    ):
        self.name = name
        self.model = ClassifierModel()
        self.logger = Logger(f'{name}_Classifier')
        self.statistics = {
            'Loss': [],
            'Precision': [],
            'Recall': [],
            'Specificity': [],
            'Accuracy': [],
            'F1': [],
            'G-Mean': [],
            'AUC': [],
        }

    def train(
            self,
            training_dataset: Dataset,
            test_dateset: Dataset,
    ) -> None:

        self.logger.info('Started training')
        # set device to cpu to speed up training precess
        former_device = config.device
        config.device = 'cpu'
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)
                label = label.to(config.device)
                prediction = self.model(x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=label,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')
        # reset to former device
        config.device = former_device

    def g_train(
            self,
            generator: nn.Module,
            training_dataset: Dataset,
            test_dateset: Dataset,
    ) -> None:

        self.logger.info('Started training with generator')
        # set device to cpu to speed up training precess
        former_device = config.device
        config.device = 'cpu'
        generator.to('cpu')
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)
                label = label.to(config.device)
                real_minority_num = int(label.sum().item())
                fake_minority_num = config.training.classifier.batch_size - real_minority_num
                z = torch.randn(fake_minority_num, config.data.z_size, device=config.device)
                supplement_x = generator(z).detach()
                balanced_x = torch.cat([x, supplement_x])
                balanced_label = torch.cat([label, torch.ones(fake_minority_num, device=config.device)])
                prediction = self.model(balanced_x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=balanced_label,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')
        # reset to former device
        config.device = former_device
        generator.to(former_device)

    def eg_train(
            self,
            encoder: nn.Module,
            generator: nn.Module,
            training_dataset: Dataset,
            test_dateset: Dataset,
            seed_dataset: Dataset,
    ) -> None:

        self.logger.info('Started training with encoder and generator')
        # set device to cpu to speed up training precess
        former_device = config.device
        config.device = 'cpu'
        encoder.to('cpu')
        generator.to('cpu')
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)

                label = label.to(config.device)
                real_minority_num = int(label.sum().item())
                fake_minority_num = len(x) - 2 * real_minority_num

                if fake_minority_num > 0:
                    seed = random.choice(seed_dataset[:][0]).to(config.device)
                    seed = torch.stack([seed for _ in range(fake_minority_num)]).to(config.device)
                    z, _, _ = encoder(seed)
                    supplement_x = generator(z).detach()
                    balanced_x = torch.cat([x, supplement_x])
                    balanced_label = torch.cat([label, torch.ones(fake_minority_num, device=config.device)])
                else:
                    balanced_x = x
                    balanced_label = label

                prediction = self.model(balanced_x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=balanced_label,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')
        # reset to former device
        config.device = former_device
        encoder.to(former_device)
        generator.to(former_device)

    def egd_train(
            self,
            encoder: nn.Module,
            generator: nn.Module,
            discriminator: nn.Module,
            training_dataset: Dataset,
            test_dateset: Dataset,
            seed_dataset: Dataset,
    ) -> None:

        self.logger.info('Started training with encoder, generator and discriminator')
        # set device to cpu to speed up training precess
        former_device = config.device
        config.device = 'cpu'
        encoder.to('cpu')
        generator.to('cpu')
        discriminator.to('cpu')
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)

                label = label.to(config.device)
                real_minority_num = int(label.sum().item())
                fake_minority_num = len(x) - 2 * real_minority_num
                majority_num = real_minority_num + fake_minority_num

                if fake_minority_num > 0:

                    seed = random.choice(seed_dataset[:][0]).to(config.device)
                    seed = torch.stack([seed for _ in range(fake_minority_num)]).to(config.device)
                    z, _, _ = encoder(seed)
                    supplement_x = generator(z).detach()
                    score = discriminator(supplement_x).detach()

                    if score.max() - score.min() > 0:
                        weight = ((score - score.min()) / (score.max() - score.min())).squeeze(dim=1)
                    else:
                        weight = torch.sigmoid(score).squeeze(dim=1)

                    weight = torch.cat([torch.ones(real_minority_num, device=config.device), weight])
                    weight = majority_num / weight.sum() * weight
                    weight = torch.cat([torch.ones(majority_num, device=config.device), weight])

                    balanced_x = torch.cat([x, supplement_x])
                    balanced_label = torch.cat([label, torch.ones(fake_minority_num, device=config.device)])
                else:
                    balanced_x = x
                    balanced_label = label
                    weight = torch.ones(len(x), device=config.device)

                prediction = self.model(balanced_x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=balanced_label,
                    weight=weight,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')
        # reset to former device
        config.device = former_device
        encoder.to(former_device)
        generator.to(former_device)
        discriminator.to(former_device)

    def _test(self, test_dataset):
        with torch.no_grad():
            self.model.eval()
            x, label = test_dataset[:]
            x = x.to(config.device)
            prob = self.model(x)
            predicted_label = self._prob2label(prob)
            tn, fp, fn, tp = confusion_matrix(
                y_true=label,
                y_pred=predicted_label,
            ).ravel()

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            specificity = tn / (tn + fp) if tn + fp != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0

            f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            g_mean = sqrt(recall * specificity)

            auc = roc_auc_score(
                y_true=label,
                y_score=predicted_label,
            )

            self.statistics['Precision'].append(precision)
            self.statistics['Recall'].append(recall)
            self.statistics['Specificity'].append(specificity)
            self.statistics['Accuracy'].append(accuracy)
            self.statistics['F1'].append(f1)
            self.statistics['G-Mean'].append(g_mean)
            self.statistics['AUC'].append(auc)
            self.model.train()

    @staticmethod
    def _prob2label(prob):
        probabilities = prob.squeeze(dim=1)
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels)

    def _plot(self):
        sns.set()
        plt.title(f"{self.name} Classifier Loss During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        sns.lineplot(data=self.statistics['Loss'])
        loss_plot_path = config.path.plots / f'{self.name}_Classifier_Loss.png'
        plt.savefig(loss_plot_path)
        plt.clf()
        self.logger.debug(f'Saved loss plot at {loss_plot_path}')

        plt.title(f"{self.name} Classifier Test Metrics During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        for name, value in self.statistics.items():
            if name == 'Loss':
                continue
            plt.plot(value, label=name)
        plt.legend()
        test_metrics_plot_path = config.path.plots / f'{self.name}_Classifier_Test_Metrics.png'
        plt.savefig(test_metrics_plot_path)
        plt.clf()
        self.logger.debug(f'Saved test metrics plot at {test_metrics_plot_path}')
