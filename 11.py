import torch
import numpy as np

import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms
cuda=torch.device('cuda:1')
a=torch.randn(2,3).to(cuda)
b=a.cpu().numpy()