from typing import Callable

import torch

from .complete_dataset import CompleteDataset


class MinorityDataset(CompleteDataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        super().__init__(training, transform, target_transform)
        target_item_indices = []

        for idx, label in enumerate(self.labels):
            if label == 1:
                target_item_indices.append(idx)
        self.features = self.features.numpy()
        self.labels = self.labels.numpy()
        self.features = self.features[target_item_indices]
        self.labels = self.labels[target_item_indices]
        self.features = torch.from_numpy(self.features)
        self.labels = torch.from_numpy(self.labels)
