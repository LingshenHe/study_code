import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
q   `
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

'''hyperparameters'''

epoch=350
batchsize=128
cuda=torch.device('cuda:5')
net=ResNet50().to(cuda)
decay=0.3
# for par in net.parameters():
#     if(len(par.size())<2):
#         nn.init.constant_(par,0)
#     else:
#         nn.init.kaiming_uniform_(par)
#

trainset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=True,transform=transforms.ToTensor()
                                    ,download=False)
testset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=False,transform=transforms.ToTensor(),download=False)
train_loader=torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True)
test_loader=torch.utils.data.DataLoader(testset,batch_size=batchsize)
train_loader1=torch.utils.data.DataLoader(trainset,batch_size=int(len(trainset)),shuffle=True)
test_loader1=torch.utils.data.DataLoader(testset,batch_size=int(len(testset)))
for times in range(epoch):
    if times<150:
        learning_rate=0.1
    elif times<250:
        learning_rate=0.01
    else:
        learning_rate=0.001
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=decay
                                )
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


