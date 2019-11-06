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

import data
import net
from client import Client

def FederatedTrain(args):

    if args.dataset == 'MNIST':
        dataset = data.LoadMNIST()
        n_channels = 1
    else:
        print 'Dataset Is Not Supported'
        exit(1)

    n_clients = args.n_clients

    n_train = dataset['x_train'].shape[0]
    n_test = dataset['x_test'].shape[0]

    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_train_split = np.split(x_train, n_clients, axis=0)
    y_train_split = np.split(y_train, n_clients, axis=0)
    x_test = dataset['x_test']
    y_test = dataset['y_test']
    tsf_train = dataset['tsf']['train']
    tsf_test = dataset['tsf']['test']

    print "x_train.shape ", x_train.shape
    print "y_train.shape ", y_train.shape
    print "x_test.shape ", x_test.shape
    print "y_test.shape ", y_test.shape
    print "len(x_train_split)", len(x_train_split)
    print "len(y_train_split)", len(y_train_split)

    global_net = net.LeNet(n_channels = n_channels)
    if args.gpu:
        global_net = global_net.cuda()

    print global_net
    print global_net.state_dict

    optimizer = optim.SGD(global_net.parameters(), lr = args.lr)

    model_dir = args.model_dir
    global_model_name = args.global_model_name

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(global_net.state_dict(), model_dir + global_model_name)
    print "Model saved"

    client = Client(net = global_net)

    for t in range(args.epochs):
        client.load_model(model_path = model_dir + global_model_name)

        for idx in range(n_clients):
            x_train_local = x_train_split[idx]
            y_train_local = y_train_split[idx]
            if args.gpu:
                x_train_local = x_train_local.cuda()
                y_train_local = y_train_local.cuda()


if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'MNIST')
        parser.add_argument('--network', type = str, default = 'LeNet')
        parser.add_argument('--n_clients', type = int, default = 4)
        parser.add_argument('--epochs', type = int, default = 200)
        parser.add_argument('--lr', type = float, default = 1e-3)
        parser.add_argument('--model_dir', type = str, default = "checkpoints/")
        parser.add_argument('--global_model_name', type = str, default = 'global_model.pth')

        parser.add_argument('--gpu', dest='gpu', action='store_true')
        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()
        assert args.n_clients > 2

        FederatedTrain(args)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
