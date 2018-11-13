import torch
import numpy as np
import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
c=1
classes=68
epoch=10
learning_rate=0.001
lamb=0.005
load_data=sio.loadmat('/home/lshe/code_study/PIE_32x32.mat')
fea=load_data['fea']
gnd=load_data['gnd']
fea=fea.astype(np.float64)
gnd=gnd.astype(np.float64)
train_size=40
a=np.zeros(69,dtype=int)
a[0]=0
j=1
for i in range(fea.shape[0]):
    if(gnd[i]>j):
        a[j]=i
        j=j+1
a[68]=fea.shape[0]
x1=np.zeros((train_size*68,1024))
y1=np.zeros((train_size*68,68))

x2=np.zeros((fea.shape[0]-train_size*68,1024))
y2=np.zeros((fea.shape[0]-train_size*68,68))


j=0
for i in range(68):
    f=np.random.choice(np.arange(a[i],a[i+1],1),a[i+1]-a[i])
    x1[i*train_size:(i+1)*train_size,::]=fea[f[0:train_size],::]
    y1[i*train_size:(i+1)*train_size,i]=1
    x2[j:j+a[i+1]-a[i]-train_size,::]=fea[f[train_size::],::]
    y2[j:j+a[i+1]-a[i]-train_size,i]=1
    j=j+a[i+1]-a[i]-train_size


x1=x1.reshape(-1,1,32,32)
x2=x2.reshape(-1,1,32,32)

class myconv(nn.Module):
    def __init__(self,c,classes):
        super(myconv,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(c,6,3,padding=1),
            nn.Conv2d(6,11,1),
            nn.Conv2d(11,5,1),
            nn.Softsign(),
            nn.Conv2d(5,1,3,padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.Conv2d(6, 11, 1),
            nn.Conv2d(11, 5, 1),
            nn.Softsign(),
            nn.Conv2d(5, 1, 3, padding=1)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(1,6,3,padding=1),
            nn.Conv2d(6,11,1),
            nn.Conv2d(11,5,1),
            nn.Softsign(),
            nn.Conv2d(5,1,3,padding=1)
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(1,6,3,padding=1),
            nn.Conv2d(6,11,1),
            nn.Conv2d(11,5,1),
            nn.Softsign(),
            nn.Conv2d(5,1,3,padding=1)
        )
        self.layer5=nn.Sequential(
            nn.Conv2d(1,6,3,padding=1),
            nn.Conv2d(6,11,1),
            nn.Conv2d(11,5,1),
            nn.Softsign(),
            nn.Conv2d(5,1,3,padding=1)
        )
    def forward(self, x):
        x=self.layer1(x)+x
        x=self.layer2(x)+x
        x=self.layer3(x)+x
        x=self.layer4(x)+x
        x=self.layer5(x)+x
        return x

conv=myconv(1,68)
# for i in conv.parameters():
#     if(len(i.size())<2):
#         nn.init.constant_(i,0)
#     else:
#         nn.init.xavier_normal_(i)
class pietrainset(torch.utils.data.Dataset):

    def __init__(self):
        self.x = x1
        self.x/=200.0
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = y1
        self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size()[0]

class  pietestset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = x2
        self.x/=200.0
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = y2
        self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size()[0]

trainset=pietrainset()
size=int(len(trainset)/2)
train_loader=torch.utils.data.DataLoader(trainset,batch_size=size
                                         ,shuffle=True)
testset=pietestset()
test_loader=torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False)

optimizer=torch.optim.SGD(conv.parameters(),lr=learning_rate)
for i in range(epoch):
    for x_train,y_train in train_loader:
        n_x=x_train.size()[0]

        xx=conv(x_train)
        # xx=x_train
        x = torch.zeros(n_x, xx.reshape(n_x, -1).size()[1] + 1)
        kkk = xx.reshape(n_x, -1).size()[1]
        x[::, 0:kkk] = xx.reshape(n_x, -1)
        x[::, kkk] = 1
        with torch.no_grad():
            W = torch.matmul(y_train.transpose(0, 1), x)
            W = torch.matmul(W, torch.inverse(torch.matmul(x.transpose(0, 1), x) + lamb * n_x * torch.eye(x.size()[1])))
        loss=torch.sum((torch.matmul(W,x.transpose(0,1))-y_train.transpose(0,1))*(torch.matmul(W,x.transpose(0,1))-y_train.transpose(0,1)))/n_x+lamb*torch.sum(W*W)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_test=torch.tensor(0.0)
    with torch.no_grad():
        for x_test,y_test in  test_loader:
            test_num=x_test.size()[0]
            x=conv(x_test)
            # x=x_test
            x=x.reshape(test_num,-1)
            xx=torch.zeros(x.size()[0],x.size()[1]+1)
            xx[::,0:x.size()[1]]=x
            xx[::,x.size()[1]]=1
            pred=torch.matmul(W,xx.transpose(0,1))
            pred=torch.argmax(pred,dim=0)
            y=torch.argmax(y_test,dim=1)
            acc_test+=torch.sum(pred==y).to(torch.float32)
            # print(acc_test)
    acc_test=acc_test/len(testset)
    acc_train = torch.tensor(0.0,dtype=torch.float32)
    with torch.no_grad():
        for x_test, y_test in train_loader:
            test_num = x_test.size()[0]
            x=conv(x_test)
            # x = x_test
            x=x.reshape(test_num, -1)
            xx=torch.zeros(test_num,x.size()[1]+1)
            xx[::,0:x.size()[1]]=x
            xx[::,x.size()[1]]=1
            pred = torch.matmul(W, xx.transpose(0, 1))
            pred = torch.argmax(pred, dim=0)
            y = torch.argmax(y_test, dim=1)
            acc_train += torch.sum(pred == y).to(torch.float32)
    acc_train=acc_train/len(trainset)
    print('In the {}/{},the loss is {},the test acc is {},the train acc is {}'.format(i,epoch,loss.data,acc_test,acc_train))





