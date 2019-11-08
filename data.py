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
from torch.utils.data import Subset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

def load_mnist():
    DATASET = 'MNIST'

    print "DATASET: ", DATASET
    if DATASET == 'MNIST':

        mu = torch.tensor((0.5,), dtype=torch.float32)
        sigma = torch.tensor((0.5,), dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

        tsf = {
            'transform': transforms.Compose([
                transforms.ToTensor(),
                Normalize
            ]),

            'target_transform': None
        }

        trainset = torchvision.datasets.MNIST(
            root = './data/MNIST',
            train = True,
            download = True,
            transform = tsf['transform'],
            target_transform = tsf['target_transform']
        )

        testset = torchvision.datasets.MNIST(
            root='./data/MNIST',
            train=False,
            download=True,
            transform = tsf['transform'],
            target_transform = tsf['target_transform']
        )

    dataset_total = {
        'train': trainset,
        'test': testset,
        'tsf': tsf
    }

    return dataset_total


class FederatedDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

        assert len(self.data) == len(self.target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.target[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def create_split_dataloaders(dataset, args):

    n_clients = args.n_clients
    n_train = len(dataset['train'].data)
    n_train_partition = n_train / n_clients

    dataloaders_train = [
        torch.utils.data.DataLoader(
            Subset(
                dataset = dataset['train'],
                indices = range(i*n_train_partition, (i+1)*n_train_partition)
            ),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = 1
        )
        for i in range(n_clients)
    ]

    dataloader_test = torch.utils.data.DataLoader(
        dataset = dataset['test'],
        batch_size = 1000,
        shuffle = False,
        num_workers = 1
    )

    print "dataset['train'].data.shape ", dataset['train'].data.shape
    print "dataset['train'].target.shape ", dataset['train'].data.shape
    print "dataset['test'].data.shape ", dataset['test'].data.shape
    print "dataset['test'].data.shape ", dataset['test'].data.shape

    for idx, loader in enumerate(dataloaders_train):
        print "Client {idx} train subsets: {l}".format(
            idx = idx,
            l = len(loader.dataset)
        )

    return dataloaders_train, dataloader_test
