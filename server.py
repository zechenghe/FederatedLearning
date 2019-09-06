import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import collections
import client
import data

class ServerNN(object):
    def __init__(n_client=2):
        super(ClientNN, self).__init__()

        self.n_client = n_client
        self.sender = client.ClientNN()
        self.receiver = client.ClientNN()
        self.clients = [self.sender, self.receiver]
        for i in range(self.n_client-2):
            self.clients.append(client.ClientNN())

        dataset_total = LoadMNIST()
        _SetClientData(dataset_total, self.clients)

    def _SetClientData(data, clients):
        n_clients = len(clients)
        n_train = data['x_train'].shape[0]
        n_test = data['x_test'].shape[0]

        n_train_share = int(n_train/n_clients)
        n_test_share = int(n_test/n_clients)

        x_train_split = np.split(data['x_train'], n_clients, axis=0)
        y_train_split = np.split(data['y_train'], n_clients, axis=0)
        x_test_split = np.split(data['x_test'], n_clients, axis=0)
        y_test_split = np.split(data['y_test'], n_clients, axis=0)

        for i in range(n_clients):
            clients[i].SetData(
                x_train = x_train_split[i],
                y_train = y_train_split[i],
                x_test = x_test_split[i],
                y_test = y_test_split[i]
            )
