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


class nnet(nn.Module):
    def __init__(self):
        super(nnet,self).__init__()
        self.conv1=nn.Conv2d(3,3,3,padding=1)
        self.bn=nn.BatchNorm2d(3)
    def forward(self,x):
        x=F.relu(self.bn(self.conv1(x)))
        return x

net1=nnet()
net2=nnet()
for pa in net1.parameters():
    pa.requires_grad=False

epoch = 350
batchsize = 128
cuda = torch.device('cuda:9')
net = pde().to(cuda)
momentum_ = 0.9
decay_ = 0.005

# for par in net.parameters():
#     if(len(par.size())<2):
#         nn.init.constant_(par,0)
#     else:
#         nn.init.kaiming_uniform_(par)
#

transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])

transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])
trainset = torchvision.datasets.CIFAR10(root='/home/lshe/code_study/', train=True, transform=transform_train
                                        , download=False)
testset = torchvision.datasets.CIFAR10(root='/home/lshe/code_study/', train=False, transform=transform_test,
                                       download=False)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize)
train_loader1 = torch.utils.data.DataLoader(trainset, batch_size=int(len(trainset)), shuffle=True)
test_loader1 = torch.utils.data.DataLoader(testset, batch_size=int(len(testset)))

for times in range(epoch):
    if times < 50:
        learning_rate = 0.02
        decay_ = 0
    elif times < 100:
        learning_rate = 0.01
        dacay_ = 0.01
    else:
        learning_rate = 0.001
        decay_ = 0.015
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), weight_decay=decay_, momentum=momentum_, lr=learning_rate)
    sum_ = torch.tensor(0.0, device=cuda)
    sum_test = torch.tensor(0.0, device=cuda)
    for x_train, y_train in train_loader:
        x_train = x_train.to(cuda)
        y_train = y_train.to(cuda)

        loss = torch.sum((net1(x)-net1(net2(x)))**2)
        optimizer.zero_grad()
        loss.backward()
        # print(loss.data)
        optimizer.step()

    net.eval()
    with torch.no_grad():
        for x_test, y_test in train_loader:
            x_test = x_test.to(cuda)
            y_test = y_test.to(cud
            sum_ += torch.sum(pred == y_test).to(torch.float32)
    print(len(trainset))
    sum_ = sum_ / len(trainset)
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(cuda)
            y_test = y_test.to(cuda)
            pred = net(x_test)
            pred = pred.argmax(dim=1)
            sum_test += torch.sum(pred == y_test).to(torch.float32)
            # print(sum_test)
    sum_test = sum_test / len(testset)

    print('the test accuracy of {}/{} iteration is {},the train accuracy is {}'.format(times, epoch, sum_test.data,sum_.data
                                                                                       ))




