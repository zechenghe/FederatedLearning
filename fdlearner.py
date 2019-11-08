# @Author: Zecheng He
# @Date:   2019-11-06T14:28:07-05:00

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

class FederatedLearner(object):
    def __init__(self, net, args):
        super(FederatedLearner, self).__init__()

        self.net = net
        self.n_clients = args.n_clients
        self.model_state = [None] * self.n_clients

        #self.optimizer = optim.SGD(self.net.parameters(), lr = args.lr)
        self.optimizer = optim.Adam(
            params = net.parameters(),
            lr = args.lr,
            eps = args.eps,
            amsgrad = args.AMSGrad
        )
        self._gpu = False

    @property
    def gpu(self):
        return self._gpu

    @gpu.setter
    def gpu(self, use_gpu):
        self._gpu = use_gpu
        if self._gpu:
            self.net = self.net.cuda()

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))

    def comp_grad(self, idx, batchX, batchY):
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        if self.gpu:
            self.net = self.net.cuda()
            batchX = batchX.cuda()
            batchY = batchY.cuda()
            criterion = criterion.cuda()
            softmax = softmax.cuda()

        optimizer.zero_grad()
        logits = self.net.forward(batchX)
        prob = softmax(logits)
        loss = criterion(logits, batchY)
        loss.backward()

        self.model_state[idx] = [x.grad.detach().cpu().numpy().copy() for x in self.net.parameters()]

    def _update_model(self):

        # Use average of gradients as global gradients
        # Update parameters according to the update rules

        self.net.zero_grad()
        for c_idx in range(self.n_clients):
            assert len(self.model_state[c_idx]) == len(list(self.net.parameters()))

            for p_idx, p in enumerate(list(self.net.parameters())):
                if self.gpu:
                    p.grad.add_(torch.tensor(self.model_state[c_idx][p_idx].cuda()))
                else:
                    p.grad.add_(torch.tensor(self.model_state[c_idx][p_idx]))

            self.optimizer.step()


    def _evalTest(self, test_loader):

        # rewind
        test_iter = iter(test_loader)

        net = self.net
        acc = 0.0
        NBatch = 0
        for i, data in enumerate(test_iter, 0):
            NBatch += 1
            batchX, batchY = data
            if self.gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            logits = net.forward(batchX)

            if self.gpu:
                pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
                groundTruth = batchY.cpu().detach().numpy()
            else:
                pred = np.argmax(logits.detach().numpy(), axis = 1)
                groundTruth = batchY.detach().numpy()
            acc += np.mean(pred == groundTruth)
        accTest = acc / NBatch
        print "Test accuracy: ", accTest
        return accTest
