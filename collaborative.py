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

def FederatedTrain(args):

    if args.dataset == 'MNIST':
        dataset = data.LoadMNIST()
    else:
        print 'Dataset Is Not Supported'
        exit(1)

    n_clients = args.n_clients

    n_train = dataset['x_train'].shape[0]
    n_test = dataset['x_test'].shape[0]

    x_train = data['x_train']
    y_train = data['y_train']
    x_train_split = np.split(x_train, n_clients, axis=0)
    y_train_split = np.split(y_train, n_clients, axis=0)
    x_test = data['x_test']
    y_test = data['y_test']
    tsf_train = data['tsf']['train']
    tsf_test = data['tsf']['test']

    print "x_train.shape ", x_train.shape
    print "y_train.shape ", y_train.shape
    print "x_test.shape ", x_test.shape
    print "y_test.shape ", y_test.shape
    print "len(x_train_split)", len(x_train_split)
    print "len(y_train_split)", len(y_train_split)

    init_net = net.LeNet()

    print init_net


if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'MNIST')
        parser.add_argument('--network', type = str, default = 'LeNet')
        parser.add_argument('--n_clients', type = int, default = 3)
        parser.add_argument('--epochs', type = int, default = 200)

        #parser.add_argument('--gpu', dest='gpu', action='store_true')
        #parser.add_argument('--nogpu', dest='gpu', action='store_false')
        #parser.set_defaults(gpu=True)
        args = parser.parse_args()
        assert args.n_clients > 2

        FederatedTrain(args)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
