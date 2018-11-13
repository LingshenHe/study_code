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
lamb=10

trainset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=True,transform=transforms.ToTensor()
                                    ,download=False)
testset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=False,transform=transforms.ToTensor(),download=False)
train_loader=torch.utils.data.DataLoader(trainset,batch_size=len(trainset),shuffle=True)
test_loader=torch.utils.data.DataLoader(testset,batch_size=len(testset))
i=iter(train_loader)
j=iter(test_loader)
x_train,y_train=i.next()
x_test,y_test=j.next()
y_train=torch.zeros(y_train.size()[0],10).scatter_(1,y_train.reshape(-1,1),1)
y_test=torch.zeros(y_test.size()[0],10).scatter_(1,y_test.reshape(-1,1),1)
x_train=x_train.to(cuda).to(torch.float32).reshape(-1,1024*3).transpose(0,1)
y_train=y_train.to(cuda).to(torch.float32).transpose(0,1)
x_test=x_test.to(cuda).to(torch.float32).reshape(-1,1024*3).transpose(0,1)
y_test=y_test.to(cuda).to(torch.float32).transpose(0,1)
xx=torch.zeros(x_train.size()[0]+1,x_train.size()[1],device=cuda)
xx[0:x_train.size()[0],::]=x_train
xx[x_train.size()[0],::]=1
w=torch.matmul(y_train,xx.transpose(0,1))
w=torch.matmul(w,torch.inverse((torch.matmul(xx,xx.transpose(0,1))+x_train.size()[1]*lamb*torch.eye(xx.size()[0],device=cuda))))
xx=torch.zeros(x_test.size()[0]+1,x_test.size()[1],device=cuda)
xx[x_test.size()[0],::]=1
xx[:x_test.size()[0],::]=x_test
pred=torch.matmul(w,xx)
pred=pred.argmax(dim=0)
yy=y_test.argmax(dim=0)

sum_=torch.sum(pred==yy).to(torch.float32)
print('acc is {}'.format(sum_/y_test.size()[1]))

