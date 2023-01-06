import random

import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config
from src.dataset import CompleteDataset


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def prepare_dataset(file_name: str):
    set_random_state()

    # concatenate the file path
    file_path = config.path.data / file_name

    # calculate skip rows
    skip_rows = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line[0] != '@':
                break
            else:
                skip_rows += 1

    # read raw data
    df = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)

    # partition label and feature
    label = np_array[:, -1].copy()
    feature = np_array[:, :-1].copy()

    # digitize label
    for i, _ in enumerate(label):
        label[i] = label[i].strip()
    label[label[:] == 'positive'] = 1
    label[label[:] == 'negative'] = 0
    label = label.astype('int')

    # normalize feature
    feature = MinMaxScaler().fit_transform(feature)

    # partition training and test sets
    training_set_size = int(config.dataset.training_ratio * len(np_array))
    training_label, test_label = np.split(label, [training_set_size])
    training_feature, test_feature = np.split(feature, [training_set_size])

    # save to files
    np.save(str(config.path.data / 'training_label.npy'), training_label)
    np.save(str(config.path.data / 'training_feature.npy'), training_feature)
    np.save(str(config.path.data / 'test_label.npy'), test_label)
    np.save(str(config.path.data / 'test_feature.npy'), test_feature)

    set_x_size()


def set_x_size():
    config.data.x_size = len(CompleteDataset()[0][0])


def get_final_test_metrics(statistics: dict):
    metrics = dict()
    for name, values in statistics.items():
        if name == 'Loss':
            continue
        else:
            metrics[name] = values[-1]
    return metrics


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())

if __name__ == '__main__':
    prepare_dataset('yeast1.dat')