# @Author: Zecheng He
# @Date:   2019-11-04T14:28:07-05:00

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
from fdlearner import FederatedLearner
from utils import *

def FederatedTrain(args):

    if args.dataset == 'MNIST':
        dataset = data.load_mnist()
        dataloaders_train, dataloader_test = data.create_split_dataloaders(
            dataset = dataset,
            args=args
        )
        dataiters_train = [iter(loader) for loader in dataloaders_train]
        dataiters_test = iter(dataloader_test)
        n_channels = 1
    else:
        print 'Dataset Is Not Supported'
        exit(1)


    n_clients = args.n_clients

    global_net = net.LeNet(n_channels = n_channels)
    print global_net

    learner = FederatedLearner(net = global_net, args = args)
    learner.gpu = args.gpu

    model_dir = args.model_dir
    global_model_name = args.global_model_name
    global_optim_name = args.global_optimizor_name
    global_model_suffix = global_model_suffix = '_init_.pth'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(learner.net.state_dict(), model_dir + global_model_name + global_model_suffix)
    print "Model saved"

    for t in range(args.epochs):
        if t == 0:
            global_model_suffix = '_init_.pth'
        else:
            global_model_suffix = '_{cur}.pth'.format(cur=t-1)

        learner.load_model(model_path = model_dir + global_model_name + global_model_suffix)

        for i in range(n_clients):
            print 't=', t, 'client model idx=', i
            try:
                batchX, batchY = next(dataiters_train[i])
            except StopIteration:
                dataiters_train[i] = iter(dataloaders_train[i])
                batchX, batchY = next(dataiters_train[i])

            learner.comp_grad(i, batchX, batchY)
        learner._update_model()

        global_model_suffix = '_{cur}.pth'.format(cur=t)
        torch.save(learner.net.state_dict(), model_dir + global_model_name + global_model_suffix)

        if (t+1) % args.n_eval_iters == 0:
            learner._evalTest(test_loader = dataloader_test)

if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'MNIST')
        parser.add_argument('--network', type = str, default = 'LeNet')
        parser.add_argument('--n_clients', type = int, default = 4)
        parser.add_argument('--epochs', type = int, default = 5000)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--lr', type = float, default = 1e-3)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--n_eval_iters', type = int, default = 1000)
        parser.add_argument('--model_dir', type = str, default = "checkpoints/")
        parser.add_argument('--global_model_name', type = str, default = 'global_model')
        parser.add_argument('--global_optimizor_name', type = str, default = 'global_optim')

        parser.add_argument('--gpu', dest='gpu', action='store_true')
        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()
        #assert args.n_clients > 1

        FederatedTrain(args)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
