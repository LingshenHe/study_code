import torch
import numpy as np

import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms

def diff(x):


    x_=torch.zeros(x.size()[0],x.size()[1]+2,x.size()[2]+2).to(device=cuda)
    x_[::,1:x.size()[1]+1,1:x.size()[2]+1]=x
    p_x=(x_[::,2:,1:x.size()[2]+1]-x_[::,:x.size()[1],1:x.size()[2]+1])/2
    p_y=(x_[::,1:x.size()[1]+1,2:]-x_[::,1:x.size()[1]+1,:x.size()[2]])/2
    pp_x=(x_[::,2:,1:x.size()[2]+1]+x_[::,:x.size()[1],1:x.size()[2]+1]-2*x)
    pp_y=(x_[::,1:x.size()[1]+1,2:]+x_[::,1:x.size()[1]+1,:x.size()[2]]-2*x)
    pp_xy=(x_[::,2:,2:]+x_[::,:x.size()[1],:x.size()[2]]-x_[::,2:,1:x.size()[2]+1]
           -x_[::,1:x.size()[1]+1,2:])
    return p_x,p_y,pp_x,pp_y,pp_xy
class pdeblock(nn.Module):
    def __init__(self,channels):
        super(pdeblock,self).__init__()
        self.conv1=nn.Conv2d(6,channels,kernel_size=1,stride=1)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,1,kernel_size=1,stride=1)
        self.bn2=nn.BatchNorm2d(1)


    def forward(self,x):
        batch=x.size()[0]
        xx=torch.zeros(batch,6,x.size()[1],x.size()[2],device=cuda)

        xx[::, 0, ::, ::]=x
        xx[::,1,::,::],xx[::,2,::,::],xx[::,3,::,::],xx[::,4,::,::],xx[::,5,::,::]=diff(x)
        xx=F.relu(self.bn1(self.conv1(xx)))
        xx=F.relu(self.bn2(self.conv2(xx)))[::,0,::,::]
        xx=xx+x
        return xx

class pdenn(nn.Module):
    def __init__(self,step,channels):
        super(pdenn,self).__init__()
        self.cha=channels
        self.layer=self.make_layer(step)


    def make_layer(self,step):
        s=[]
        for i in range(step):
            s.append(pdeblock(self.cha))
        return nn.Sequential(*s)

    def forward(self, x):
        return self.layer(x)

class lpde_(nn.Module):
    def __init__(self,eqnum,step,channels):
        super(lpde_,self).__init__()
        self.eq=eqnum
        self.list=nn.ModuleList([pdenn(step,channels) for i in range(eqnum)])
    def forward(self, x):
        xx=torch.zeros(x.size()[0],eqnum,x.size()[1],x.size()[2],device=cuda)
        for i in range(self.eq):
            xx[::,i,::,::]=self.list[i](x)
        return xx

class lpde(nn.Module):
    def __init__(self,eqnum,step,channels):
        super(lpde,self).__init__()
        self.eq=eqnum
        self.lpde=lpde_(eqnum,step,channels)
        self.linear=nn.Linear(eqnum*3*64,10)
    def forward(self,x):
        xx=torch.zeros(x.size()[0],self.eq*3,x.size()[2],x.size()[3],device=cuda)
        for i in range(3):
            xx[::,i*self.eq:(i+1)*self.eq,::,::]=self.lpde(x[::,i,::,::])
        xx=F.avg_pool2d(xx,4)
        xx=xx.reshape(xx.size()[0],-1)
        xx=self.linear(xx)
        return xx
'''hyperparameter'''
eqnum=10
step=10
decay_=0.003
epoch=350
channels=25
batchsize=64
momentum_=0.9
cuda=torch.device('cuda:4')
net=lpde(eqnum,step,channels)
net=net.to(cuda)






trainset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=True,transform=transforms.ToTensor()
                                    ,download=False)
testset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=False,transform=transforms.ToTensor(),download=False)
train_loader=torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True)
test_loader=torch.utils.data.DataLoader(testset,batch_size=batchsize)
train_loader1=torch.utils.data.DataLoader(trainset,batch_size=int(len(trainset)),shuffle=True)
test_loader1=torch.utils.data.DataLoader(testset,batch_size=int(len(testset)))

for times in range(epoch):
    if times<50:
        learning_rate=0.1
    elif times<100:
        learning_rate=0.01
    else:
        learning_rate=0.001
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=decay_,momentum=momentum_)
    sum_=torch.tensor(0.0,device=cuda)
    sum_test=torch.tensor(0.0,device=cuda)
    for x_train,y_train in train_loader:
        x_train=x_train.to(cuda)
        y_train=y_train.to(cuda)

        loss=nn.functional.cross_entropy(net(x_train),y_train)
        optimizer.zero_grad()
        loss.backward()
        # print(loss.data)
        optimizer.step()

    net.eval()
    with torch.no_grad():
        for x_test,y_test in train_loader:
            x_test=x_test.to(cuda)
            y_test=y_test.to(cuda)
            pred=net(x_test)
            pred=pred.argmax(dim=1)
            sum_+=torch.sum(pred==y_test).to(torch.float32)
    sum_=sum_/len(trainset)
    with torch.no_grad():
        for x_test,y_test in test_loader:
            x_test=x_test.to(cuda)
            y_test=y_test.to(cuda)
            pred=net(x_test)
            pred=pred.argmax(dim=1)
            sum_test+=torch.sum(pred==y_test).to(torch.float32)
            # print(sum_test)
    sum_test=sum_test/len(testset)

    print('the test accuracy of {}/{} iteration is {},the train accuracy is {}'.format(times,epoch,sum_.data,sum_test.data))


