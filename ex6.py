import torch
import numpy as np


import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms


class block(nn.Module):
    c = 7

    conv= torch.nn.Conv2d(3 , 18 , 3, stride=1, padding=1, bias=False, groups=3)

    def __init__(self, eqnum):
        c = block.c
        super(block, self).__init__()
        self.conv1 =nn.Conv2d(6 * eqnum, c * eqnum, 1, stride=1, padding=0, groups=1)
        self.conv2 = nn.Conv2d(c * eqnum, eqnum, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(c * eqnum)
        self.bn2 = nn.BatchNorm2d(eqnum)
        # self.w=nn.Parameter(0.2*torch.rand(1))

    def forward(self, x):
        y=block.conv(x)
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.selu(self.bn2(self.conv2(y)))
        return x + y

class pde(nn.Module):
    steps = 10
    eqnum = 32

    def __init__(self):
        super(pde, self).__init__()
        self.conv=nn.Conv2d(3,pde.eqnum,3,stride=2,padding=1)
        self.bn=nn.BatchNorm2d(pde.eqnum)
        self.linear = nn.Linear(16 * 16*pde.eqnum, 10)
        block.conv=torch.nn.Conv2d(pde.eqnum , 6*pde.eqnum , 3, stride=1, padding=1, bias=False, groups=pde.eqnum).to(device=cuda)

        for i in block.conv.parameters():
            i.requires_grad = False

        b = torch.zeros(6, 3, 3).to(device=cuda)
        b[0, ::, ::] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(device=cuda)

        b[1, ::, ::] = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]).to(device=cuda)
        b[2, ::, ::] = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]]).to(device=cuda)
        b[3, ::, ::] = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]]).to(device=cuda)
        b[4, ::, ::] = b[3, ::, ::].transpose(0, 1).to(device=cuda)
        b[5, ::, ::] = torch.tensor([[1, -1, 0], [-1, 2, -1], [0, -1, 1]]).to(device=cuda)

        for i in range(pde.eqnum):
            block.conv.weight.data[i*6:i*6+6, 0, ::, ::] = b
        # self.layer1=block(pde.eqnum)
        # self.layer2=block(pde.eqnum)
        # self.layer3=block(pde.eqnum)
        # self.layer4=block(pde.eqnum)
        # self.layer5=block(pde.eqnum)
        # self.layer6=block(pde.eqnum)
        # self.layer7=block(pde.eqnum)
        # self.layer8=block(pde.eqnum)
        # self.layer9=block(pde.eqnum)
        # self.layer10=block(pde.eqnum)
        self.layer=nn.ModuleList([block(pde.eqnum) for i in range(pde.steps)])


    def make(self, st):
        a = []
        for i in range(st):
            a.append(block(pde.eqnum))
        return nn.Sequential(*a)

    def forward(self, x):
        y=F.relu(self.bn(self.conv(x)))
        # y=self.layer1(y)
        # y=self.layer2(y)
        # y=self.layer3(y)
        # y=self.layer4(y)
        # y=self.layer5(y)
        # y=self.layer6(y)
        # y=self.layer7(y)
        # y=self.layer8(y)
        # y=self.layer9(y)
        # y=self.layer10(y)
        # y=self.layer9(y)
        # y=self.layer5(y)
        # y = F.avg_pool2d(y, 4)
        for i in range(pde.steps):
            y=self.layer[i](y)
        y=y.reshape(y.size(0),-1)


        return self.linear(y)

block.c=5
pde.steps=7
pde.eqnum=32
batchsize = 128

epoch = 205
cuda = torch.device('cuda:6')
net = pde().to(cuda)
momentum_ = 0.90
decay_ = 0.005
op=0.8



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
    elif times <200:
        learning_rate = 0.001
        decay_ = 0.015
    elif times<250:
        learning_rate=0.0001
        decay_=0.01
    else:
        learning_rate=0.00005

        decay_=0.0001
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), weight_decay=decay_, momentum=momentum_, lr=learning_rate)
    sum_ = torch.tensor(0.0, device=cuda)
    sum_test = torch.tensor(0.0, device=cuda)
    for x_train, y_train in train_loader:
        x_train = x_train.to(cuda)
        y_train = y_train.to(cuda)

        loss = nn.functional.cross_entropy(net(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        # print(loss.data)
        optimizer.step()

    net.eval()
    with torch.no_grad():
        for x_test, y_test in train_loader:
            x_test = x_test.to(cuda)
            y_test = y_test.to(cuda)
            pred = net(x_test)
            pred = pred.argmax(dim=1)
            sum_ += torch.sum(pred == y_test).to(torch.float32)
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
    if sum_test>op:
        op=sum_test
        torch.save(net.state_dict(), '00real_step{}_c{}_eqnum{}batchsize{}'.format(pde.steps,block.c,pde.eqnum,batchsize))
    print('00real c={},step={},eqnum={},batchsize={}'.format(block.c, pde.steps, pde.eqnum, batchsize))

    print('the test accuracy of {}/{} iteration is {},the train accuracy is {}'.format(times, epoch, sum_test.data,sum_.data
                                                                                       ))

