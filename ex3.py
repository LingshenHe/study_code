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
from tensorboardX import SummaryWriter

import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms



class block(nn.Module):
    c = 5

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
    steps = 2
    eqnum = 35

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
            block.conv.weight.data[i:i + 6, 0, ::, ::] = b
        self.layer1=block(pde.eqnum)
        self.layer2=block(pde.eqnum)
        self.layer3=block(pde.eqnum)
        self.layer4=block(pde.eqnum)
        self.layer5=block(pde.eqnum)
        self.layer6=block(pde.eqnum)
        self.layer7=block(pde.eqnum)
        self.layer8=block(pde.eqnum)
        self.layer9=block(pde.eqnum)
        # self.layer10=block(pde.eqnum)


    def make(self, st):
        a = []
        for i in range(st):
            a.append(block(pde.eqnum))
        return nn.Sequential(*a)

    def forward(self, x):
        y=F.relu(self.bn(self.conv(x)))
        y=self.layer1(y)
        y=self.layer2(y)
        y=self.layer3(y)
        y=self.layer4(y)
        y=self.layer5(y)
        y=self.layer6(y)
        y=self.layer7(y)
        y=self.layer8(y)
        y=self.layer9(y)
        y=self.layer9(y)
        # y=self.layer5(y)
        # y = F.avg_pool2d(y, 4)
        y=y.reshape(y.size(0),-1)


        return self.linear(y)
cuda=torch.device('cuda:8')
writer=SummaryWriter('pic')
net=pde().to(cuda)

net.load_state_dict(torch.load('step9_eq_35_c5'))

transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])

testset = torchvision.datasets.CIFAR10(root='/home/lshe/code_study/', train=False, transform=transform_test,
                                       download=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128)

for x_test,y_test in test_loader:
    x_test=x_test.to(cuda)
    x_test.requires_grad=True
writer.add_graph(net,x_test)
y1=F.relu(net.bn(net.conv(x_test)))

y2=net.layer1(y1)
y3=net.layer2(y2)
y4=net.layer3(y3)
y5=net.layer4(y4)
y6=net.layer5(y5)
y7=net.layer6(y6)
y8=net.layer7(y7)
y9=net.layer8(y8)
y10=net.layer9(y9)
y11=net.layer9(y10)
writer.add_image('0',x_test[0][0:3])
writer.add_image('2',y2[0][0])
writer.add_image('3',y3[0][0])
writer.add_image('4',y4[0][0])
writer.add_image('5',y5[0][0])
writer.add_image('6',y6[0][0])
writer.add_image('7',y7[0][0])
writer.add_image('8',y8[0][0])
writer.add_image('9',y9[0][0])
writer.add_image('10',y10[0][0])
writer.add_image('11',y11[0][0])



# net(x_test)

writer.close()


