import torch
import numpy as np


import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms

import torch
import numpy as np

import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class mylu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, k1, k2, k3, b, bias=None):
        ctx.save_for_backward(input, k1, k2, k3, b, bias)

        output = k1 * input * (input < 0).to(torch.float32) + k2 * input * ((input >= 0) * (input < b)).to(
            torch.float32) + (k2 * b + (input - b) * k3) * (input >= b).to(torch.float32)
        if bias is not None:
            return output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, k1, k2, k3, b, bias = ctx.saved_tensors
        grad_input = gradk1 = gradk2 = gradk3 = grad_b = grad_bias = None
        grad_input = (k1 * (input < 0).to(torch.float32) + k2 * ((input >= 0) * (input < b)).to(torch.float32) + k3 * (
                    input >= b).to(torch.float32)) * grad_output
        gradk1 = torch.sum(input * (input < 0).to(torch.float32) * grad_output)
        gradk2 = torch.sum(
            (input * ((input >= 0) * (input < b)).to(torch.float32) + b * (input >= b).to(torch.float32)) * grad_output)
        gradk3 = torch.sum(b * (input >= b).to(torch.float32) * grad_output)
        grad_b = torch.sum((k2 - k3) * (x > b).to(torch.float32))
        # print(grad_input, gradk1, gradk2, gradk3, grad_b, grad_bias)
        if bias is not None:
            grad_bias = torch.sum(grad_output)
        return grad_input, gradk1, gradk2, gradk3, grad_b, grad_bias


class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None


f = mylu.apply
f1 = MulConstant.apply
x = torch.arange(-5, 7, 0.01)
k_ = torch.tensor([1, 3, 6, 4]).to(torch.float32)
y = f(x, k_[0], k_[1], k_[2], k_[3])
print('enen')
k = torch.tensor([0.1, 0.0, 0.0, 1.0], requires_grad=True)

learning_rate = 0.00005
for i in range(1000):
    if(i==640):
        learning_rate*=0.001
    pred = f(x, k[0], k[1], k[2], k[3])
    loss = torch.sum((pred - y) * (pred - y))
    print('The loss of {} iteration is {}'.format(i,loss))
    loss.backward()

    with torch.no_grad():
        k -= learning_rate * k.grad
    k.grad.zero_()
print(k)
