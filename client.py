import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import collections

class Client(object):
    def __init__(self, net, args):
        super(Client, self).__init__()

        self.net = net
        self.n_clients = args.n_clients
        self.model_state = [None] * self.n_clients

        self.optimizer = optim.SGD(net.parameters(), lr = args.lr)
        #self.optimizer = optim.Adam(
        #        params = net.parameters(),
        #        lr = args.lr,
        #        eps = args.eps,
        #        amsgrad = args.AMSGrad
        #    )
        self.gpu = False

    @property
    def gpu(self):
        return self.gpu

    @gpu.setter
    def gpu(self, gpu):
        self.gpu = gpu

    def load_model(model_path):
        self.net.load_state_dict(torch.load(model_path))

    def comp_grad(self, idx, batchX, batchY):
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        if sels.gpu:
            self.net = self.net.cuda()
            batchX = batchX.cuda()
            batchY = batchY.cuda()
            optimizer = optimizer.cuda()
            criterion = criterion.cuda()
            softmax = softmax.cuda()

        optimizer.zero_grad()
        logits = self.net.forward(batchX)
        prob = softmax(logits)
        loss = criterion(logits, batchY)
        loss.backward()

        self.model_state[idx] = self.net.named_parameters().copy()
