import torch
import numpy as np

import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms
'''hyperparameters'''

trainset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study',train=True,transform=transforms.ToTensor()
                                    ,download=False)
testset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study',train=False,transform=transforms.ToTensor()
                                    ,download=False)

print(testset[0][0])
