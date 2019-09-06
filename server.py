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

class ServerNN(object):
    def __init__(n_client=2):
        super(ClientNN, self).__init__()

        self.n_client = n_client
        self.sender = client.ClientNN()
        self.receiver = client.ClientNN()
        self.others = []
        for i in range(self.n_client-2):
            self.others.append(client.ClientNN())

        
