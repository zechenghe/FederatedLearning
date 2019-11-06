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

class Client(object):
    def __init__(self, net):
        super(Client, self).__init__()
        self.net = net

    def load_model(self, model_path):

        self.net.load_state_dict(torch.load(model_path))
