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

from net import *

def main():

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

    print "len(trainset) ", len(trainset)
    print "len(testset) ", len(testset)
    x_train, y_train = trainset.train_data, trainset.train_labels,
    x_test, y_test = testset.test_data, testset.test_labels,

    print "x_train.shape ", x_train.shape
    print "x_test.shape ", x_test.shape

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    inverseloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)
    inverseIter = iter(inverseloader)

    


if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
