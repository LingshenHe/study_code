import torch
import numpy as np
import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
load_data=sio.loadmat('/home/lshe/code_study/PIE_32x32.mat')
fea=load_data['fea']
gnd=load_data['gnd']
fea=fea.astype(np.float64)
gnd=gnd.astype(np.float64)

cuda=torch.device('cuda:0')

epoch=80
steps=5#时间轴步数
eqnum=3#微分方程的个数
difnum=6#微分不变量的个数
learning_rate=0.5

momentum=0
lamb = 20

learning_decay=1
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
def diff(x):
    x_=torch.zeros(x.size()[0],x.size()[1]+2,x.size()[2]+2,device=cuda)
    x_[::,1:x.size()[1]+1,1:x.size()[2]+1]=x
    p_x=(x_[::,2:,1:x.size()[2]+1]-x_[::,:x.size()[1],1:x.size()[2]+1])/2
    p_y=(x_[::,1:x.size()[1]+1,2:]-x_[::,1:x.size()[1]+1,:x.size()[2]])/2
    pp_x=(x_[::,2:,1:x.size()[2]+1]+x_[::,:x.size()[1],1:x.size()[2]+1]-2*x)
    pp_y=(x_[::,1:x.size()[1]+1,2:]+x_[::,1:x.size()[1]+1,:x.size()[2]]-2*x)
    pp_xy=(x_[::,2:,2:]+x_[::,:x.size()[1],:x.size()[2]]-x_[::,2:,1:x.size()[2]+1]-x_[::,1:x.size()[1]+1,2:])
    return p_x,p_y,pp_x,pp_y,pp_xy

'''
x_train=torch.randn(64,3,20,20)#训练集输入
y_train=torch.randn(64,38)#训练集输出

'''
trainset=pietrainset()

train_loader=torch.utils.data.DataLoader(trainset,batch_size=int(len(trainset))
                                         ,shuffle=True)
testset=pietestset()
test_loader=torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False)


#---------------------------------------------------------#
#         开始训练                                        #
#---------------------------------------------------------#

# c_y=y_train.size()[1]#总共有多少类
# test_num=x_test.size()[0]

#-----------------------------------#
# lpde system（fearture extraction)#
#-----------------------------------#

a=torch.randn(eqnum,difnum,steps,device=cuda)
a.requires_grad_()


delta_a=torch.zeros_like(a)
for times in range(epoch):
    if((times+1)%20==0):
        learning_rate/=2
    for x_train,y_train in train_loader:
        x_train=x_train.to(cuda)
        y_train=y_train.to(cuda)
    # data_iter=iter(train_loader)
    # x_train,y_train=data_iter.next()
        n_x, f_x, w_x, h_x = x_train.size()

        xx = torch.zeros(n_x, eqnum, f_x, w_x, h_x,device=cuda)

        for i in range(f_x):

            for k in range(eqnum):
                tt=x_train[::,i,::,::]

                for j in range(steps):

                    p_x,p_y,p_xx,p_yy,p_xy=diff(tt)
                    tt=tt+F.softsign(a[k][0][j]+a[k][1][j]*tt+a[k][2][j]*(p_x*p_x+p_y*p_y)+a[k][3][j]*(p_xx+p_yy)+a[k][4][j]*
                                     (p_x*p_x*p_xx+2*p_x*p_y*p_xy+p_y*p_y*p_yy)+a[k][5][j]*(p_xx*p_xx+2*p_xy*p_xy+p_yy*p_yy))

                xx[::,k,i,::,::]=tt
        # xx=xx.reshape(n_x,-1)
        x=torch.zeros(n_x,xx.reshape(n_x,-1).size()[1]+1,device=cuda)
        kkk=xx.reshape(n_x,-1).size()[1]
        x[::,0:kkk]=xx.reshape(n_x,-1)
        x[::,kkk]=1



        # u=xx_.detach().numpy()
        # u=u.T
        # # print(u.shape)
        # y=y_train.numpy().T
        # w_num=np.dot(y,u.T)
        # f_=np.dot(u,u.T)+lamb*n_x*np.eye(np.size(u,0))
        #
        # fff=np.linalg.inv(f_)
        # w_num =np.dot(w_num,fff)
        # W=torch.from_numpy(w_num)
        with torch.no_grad():
            W = torch.matmul(y_train.transpose(0, 1), x)
            W = torch.matmul(W, torch.inverse(torch.matmul(x.transpose(0, 1), x) + lamb * n_x * torch.eye(x.size()[1],device=cuda)))

        # W=W.to(torch.float32)


        loss=torch.sum((torch.matmul(W,x.transpose(0,1))-y_train.transpose(0,1))*(torch.matmul(W,x.transpose(0,1))-y_train.transpose(0,1)))/n_x+lamb*torch.sum(W*W)


        loss.backward()

        with torch.no_grad():
            delta_a = delta_a * momentum + (1 - momentum) * a.grad
            a-=learning_rate*delta_a
            learning_rate=learning_decay*learning_rate
        a.grad.zero_()

    # test

    with torch.no_grad():
        sum_=torch.tensor(0,dtype=torch.float32,device=cuda)
        #for x_test,y_test in test_loader:
        for x_test,y_test in test_loader:
            # print(y_test.size())
            x_test=x_test.to(cuda)
            y_test=y_test.to(cuda)

            test_num, f_x, w_x, h_x = x_test.size()

            xx = torch.zeros(test_num, eqnum, f_x, w_x, h_x,device=cuda)

            for i in range(f_x):


                for k in range(eqnum):
                    tt = x_test[::, i, ::, ::]
                    # print(k)
                    for j in range(steps):
                        p_x, p_y, p_xx, p_yy, p_xy = diff(tt)
                        tt = tt + F.softsign(
                            a[k][0][j] + a[k][1][j] * tt + a[k][2][j] * (p_x * p_x + p_y * p_y) + a[k][3][j] * (
                                        p_xx + p_yy) + a[k][4][j] *
                            (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j] * (
                                        p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))

                    xx[::, k, i, ::, ::] = tt
            xx = xx.reshape(test_num, -1)#最后学到的feature
            xx_ = torch.zeros(test_num, xx.size()[1] + 1,device=cuda)
            xx_[::, 0:xx.size()[1]] = xx
            xx_[::, xx.size()[1]] = 1
            pred=torch.matmul(W,xx_.transpose(0,1))
            pred=torch.argmax(pred,dim=0)
            init=torch.argmax(y_test,dim=1)
            # print(pred.size())
            # print(init.size())
            sum_+=torch.sum(pred==init).to(torch.float32)
            # print(sum_)
        acc_test=sum_/len(testset)

        with torch.no_grad():
            sum_ = torch.tensor(0, dtype=torch.float32,device=cuda)
            # for x_test,y_test in test_loader:
            for x_test, y_test in train_loader:
                # print(y_test.size())
                x_test=x_test.to(cuda)
                y_test=y_test.to(cuda)
                test_num, f_x, w_x, h_x = x_test.size()

                xx = torch.zeros(test_num, eqnum, f_x, w_x, h_x,device=cuda)

                for i in range(f_x):

                    for k in range(eqnum):
                        tt = x_test[::, i, ::, ::]
                        # print(k)
                        for j in range(steps):
                            p_x, p_y, p_xx, p_yy, p_xy = diff(tt)
                            tt = tt + F.softsign(
                                a[k][0][j] + a[k][1][j] * tt + a[k][2][j] * (p_x * p_x + p_y * p_y) + a[k][3][j] * (
                                        p_xx + p_yy) + a[k][4][j] *
                                (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j] * (
                                        p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))

                        xx[::, k, i, ::, ::] = tt
                xx = xx.reshape(test_num, -1)  # 最后学到的feature
                xx_ = torch.zeros(test_num, xx.size()[1] + 1,device=cuda)
                xx_[::, 0:xx.size()[1]] = xx
                xx_[::, xx.size()[1]] = 1
                pred = torch.matmul(W, xx_.transpose(0, 1))
                pred = torch.argmax(pred, dim=0)
                init = torch.argmax(y_test, dim=1)
                # print(pred.size())
                # print(init.size())
                sum_ += torch.sum(pred == init).to(torch.float32)
                # print(sum_)
            acc_train = sum_ / len(trainset)






    # #print('the loss of {}/{} iteration is {}'.format(times,epoch,loss.data))
    print('the test accuracy of {}/{} iteration is {},the train accuracy is {}'.format(times,epoch,acc_test.data,acc_train.data))

#以上得pde系统是共享参数的
'''----------------------------------------------------
   非共享参数
    a = torch.zeros(eqnum, difnum,  steps,f_x)
    a.requires_grad_()

    delta_a = torch.zeros_like(a)
    for times in range(epoch):
        xx = torch.zeros(n_x, eqnum, f_x, w_x, h_x)
        for i in range(f_x):

            for k in range(eqnum):
                tt = x_train[::, i, ::, ::]
                print(k)
                for j in range(steps):
                    p_x, p_y, p_xx, p_yy, p_xy = diff(tt)
                    tt = tt + F.softsign(
                        a[k][0][j][i] + a[k][1][j][i] * tt + a[k][2][j][i] * (p_x * p_x + p_y * p_y) + a[k][3][j][i] * (
                                    p_xx + p_yy) + a[k][4][j][i] *
                        (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][i] * (
                                    p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))

                xx[::, k, i, ::, ::] = tt
        xx = xx.reshape(n_x, -1)
'''









#----------------------------------------------------------------


#-------------------------------------------------------------#
#                  训练结束                                    #
#-------------------------------------------------------------#





