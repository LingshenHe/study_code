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
        self.conv=nn.Conv2d(eqnum,6*eqnum,3,padding=1,bias=False)
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
    eqnum = 3

    def __init__(self):
        super(pde, self).__init__()
        self.conv=nn.Conv2d(1,pde.eqnum,3,stride=2,padding=1)
        self.bn=nn.BatchNorm2d(pde.eqnum)
        self.linear = nn.Linear(16 * 16*pde.eqnum, 68)
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

block.c=1
pde.steps=3
pde.eqnum=32
print('pie c={},step={},eqnum={}'.format(block.c,pde.steps,pde.eqnum))
epoch = 300
batchsize = 128
train_size = 20
cuda = torch.device('cuda:4')
net = pde().to(cuda)
momentum_ = 0.9
decay_ = 0.005
op=0.8



# for par in net.parameters():
#     if(len(par.size())<2):
#         nn.init.constant_(par,0)
#     else:
#         nn.init.kaiming_uniform_(par)
#

load_data = sio.loadmat('/home/lshe/code_study/PIE_32x32.mat')
fea = load_data['fea']
gnd = load_data['gnd']
fea = fea.astype(np.float64)
gnd = gnd.astype(np.float64)



a = np.zeros(69, dtype=int)
a[0] = 0
j = 1
for i in range(fea.shape[0]):
    if (gnd[i] > j):
        a[j] = i
        j = j + 1
a[68] = fea.shape[0]
x1 = np.zeros((train_size * 68, 1024))
y1 = np.zeros(train_size * 68,dtype=np.int64)

x2 = np.zeros((fea.shape[0] - train_size * 68, 1024))
y2 = np.zeros(fea.shape[0] - train_size * 68,dtype=np.int64)

j = 0
for i in range(68):
    f = np.random.choice(np.arange(a[i], a[i + 1], 1), a[i + 1] - a[i])
    x1[i * train_size:(i + 1) * train_size, ::] = fea[f[0:train_size], ::]
    y1[i * train_size:(i + 1) * train_size] = i
    x2[j:j + a[i + 1] - a[i] - train_size, ::] = fea[f[train_size::], ::]

    y2[j:j + a[i + 1] - a[i] - train_size] = i
    j = j + a[i + 1] - a[i] - train_size


class pietrainset(torch.utils.data.Dataset):

    def __init__(self):
        self.x = x1.reshape(-1, 1, 32, 32)
        self.x /= 200.0
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(y1)
        # self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size()[0]


class pietestset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = x2.reshape(-1, 1, 32, 32)
        self.x /= 200.0
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(y2)
        # self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size()[0]


trainset = pietrainset()
testset = pietestset()
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize)

for times in range(epoch):
    if times < 50:

        learning_rate=0.001
        decay_=0.05
    elif times<100:
        learning_rate=0.0005
        decay_=0.001
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
    if sum_test>op:
        op=sum_test
        torch.save(net.state_dict(), 'pie_step{}_c{}_eqnum{}'.format(pde.steps,block.c,pde.eqnum))

    print('the test accuracy of {}/{} iteration is {},the train accuracy is {}'.format(times, epoch, sum_test.data,sum_.data
                                                                                       ))

