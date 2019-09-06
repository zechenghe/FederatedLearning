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

def main():
    pass

if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
