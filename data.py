import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

def LoadMNIST():
    DATASET = 'MNIST'

    print "DATASET: ", DATASET
    if DATASET == 'MNIST':

        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

        tsf = {
            'train': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ]),

            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ])
        }

        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True,
                                        download=True, transform = tsf['train'])

        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False,
                                       download=True, transform = tsf['test'])

    x_train, y_train = trainset.train_data, trainset.train_labels,
    x_test, y_test = testset.test_data, testset.test_labels,

    dataset_total = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'tsf': tsf
    }

    return dataset_total
