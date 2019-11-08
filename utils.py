# @Author: Zecheng He
# @Date:   2019-11-06T14:28:07-05:00

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
from skimage.measure import compare_ssim as ssim

import cv2

def accuracy(predictions, labels):

    if not (predictions.shape == labels.shape):
        print "predictions.shape ", predictions.shape, "labels.shape ", labels.shape
        raise AssertionError

    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def pseudoInverse(W):
    return np.linalg.pinv(W)


def getImgByClass(Itr, C = None):

    if C == None:
        return Itr.next()

    while (True):
        img, label = Itr.next()
        if label == C:
            break
    return img, label


def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data

def preprocess(data):

    size = data.shape
    NChannels = size[-1]
    assert NChannels == 1 or NChannels == 3
    if NChannels == 1:
        mu = 0.5
        sigma = 0.5
    elif NChannels == 3:
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]
    data = (data - mu) / sigma

    assert data.shape == size
    return data


def deprocess(data):

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1:
        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    elif NChannels == 3:
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    else:
        print "Unsupported image in deprocess()"
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0,:,:,:]).unsqueeze(0))


def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def l2loss(x):
    return (x**2).mean()

def l1loss(x):
    return (torch.abs(x)).mean()


def getModule(net, blob):
    modules = blob.split('.')

    curr_module = net
    print curr_module
    for m in modules:
        curr_module = curr_module._modules.get(m)
    return curr_module

def getLayerOutputHook(module, input, output):
    if not hasattr(module, 'activations'):
        module.activations = []
    module.activations.append(output)

def getHookActs(model, module, input):
    if hasattr(module, 'activations'):
        del module.activations[:]
    _ = model.forward(input)
    assert(len(module.activations) == 1)
    return module.activations[0]

def saveHeatmap(mask, img, filepath):

    mask = mask[0,0,:,:]
    mask = 1-mask
    mask = (mask - np.min(mask)) / np.max(mask)

    heatmap = cv2.applyColorMap(np.uint8(255*(mask)), cv2.COLORMAP_JET)
    cv2.imwrite(filepath + '-heatmap.png', heatmap)

    img = np.moveaxis(img[0,:,:,:], 0, -1)
    imgHeatmap = 0.5 * heatmap + 0.5 * (img * 255)
    cv2.imwrite(filepath + '-heatmapImg.png', imgHeatmap)

def saveImage(img, filepath):
    torchvision.utils.save_image(img, filepath)

def get_PSNR(refimg, invimg, peak = 1.0):
    psnr = 10*np.log10(peak**2 / np.mean((refimg - invimg)**2))
    return psnr
