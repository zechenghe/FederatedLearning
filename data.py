# @Author: Zecheng He
# @Date:   2019-11-06T14:28:07-05:00

import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

def load_mnist():
    DATASET = 'MNIST'

    print "DATASET: ", DATASET
    if DATASET == 'MNIST':

        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

        tsf = {
            'transform': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ]),

            'target_transform': None
        }

        trainset = torchvision.datasets.MNIST(
            root='./data/MNIST',
            train=True,
            download=True
        )

        testset = torchvision.datasets.MNIST(
            root='./data/MNIST',
            train=False,
            download=True
        )

    x_train, y_train = trainset.data, trainset.targets,
    x_test, y_test = testset.data, testset.targets,

    dataset_total = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'tsf': tsf
    }

    return dataset_total


class FederatedDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transfor = transform
        self.target_transform = target_transform

        assert len(self.data) != len(self.target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def create_split_dataloaders(dataset, args):

    n_clients = args.n_clients
    n_train = dataset['x_train'].shape[0]
    n_test = dataset['x_test'].shape[0]

    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_train_split = np.split(x_train, n_clients, axis=0)
    y_train_split = np.split(y_train, n_clients, axis=0)

    dataloaders_train = [
        torch.utils.data.DataLoader(
            FederatedDataset(
                data = x_train_split[i],
                target = y_train_split[i],
                transform = dataset['tsf']['transform'],
                target_transform = dataset['tsf']['target_transform']
            ),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = 1
        )
        for i in range(n_clients)
    ]

    x_test = dataset['x_test']
    y_test = dataset['y_test']
    dataloader_test = torch.utils.data.DataLoader(
        FederatedDataset(
            data = x_test,
            target = y_test,
            transform = dataset['tsf']['transform'],
            target_transform = dataset['tsf']['target_transform']
        ),
        batch_size = 1000,
        shuffle = False,
        num_workers = 1
    )

    print "x_train.shape ", x_train.shape
    print "y_train.shape ", y_train.shape
    print "x_test.shape ", x_test.shape
    print "y_test.shape ", y_test.shape
    print "len(x_train_split)", len(x_train_split)
    print "len(y_train_split)", len(y_train_split)

    return dataloaders_train, dataloader_test
