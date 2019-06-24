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

class LeNet(nn.Module):
    def __init__(self, NChannels):
        super(LeNet, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.ReLU(True)
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act


        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

        self.fc2act = nn.ReLU(True)
        self.classifier.append(self.fc2act)
        self.layerDict['fc2act'] = self.fc2act

        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layerDict['fc3'] = self.fc3


    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x

    def forward_from(self, x, layer):

        if layer in self.layerDict:
            targetLayer = self.layerDict[layer]

            if targetLayer in self.features:
                layeridx = self.features.index(targetLayer)
                for func in self.features[layeridx+1:]:
                    x = func(x)
#                    print "x.size() ", x.size()

#                print "Pass Features "
                x = x.view(-1, self.feature_dims)
                for func in self.classifier:
                    x = func(x)
                return x

            else:
                layeridx = self.classifier.index(targetLayer)
                for func in self.classifier[layeridx+1:]:
                    x = func(x)
                return x
        else:
            print "layer not exists"
            exit(1)


    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        x = x.view(-1, self.feature_dims)
        for layer in self.classifier:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")
