from typing import Callable

import torch
import numpy as np
from torch.utils.data import Dataset as Base

from src import config


class CompleteDataset(Base):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        if training is True:
            labels_path = config.path.data / 'training_label.npy'
            features_path = config.path.data / 'training_feature.npy'
        else:
            labels_path = config.path.data / 'test_label.npy'
            features_path = config.path.data / 'test_feature.npy'

        self.features = torch.from_numpy(
            np.load(str(features_path))
        ).float()
        self.labels = torch.from_numpy(
            np.load(str(labels_path))
        ).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.features[item]
        label = self.labels[item]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label


if __name__ == '__main__':
    CompleteDataset()
