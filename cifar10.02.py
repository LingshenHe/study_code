import torch
import numpy as np
import gc
import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms
'''hyperparameters'''
'''多分支'''
batchsize=128
eqnum=5
difnum=6
steps=5
learning_rate=0.1
learning_decay=1
momentum=0
epoch=40
cuda=torch.device('cuda:2')
classes=10
lamb=5
decay=0
branch=3

trainset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=True,transform=transforms.ToTensor()
                                    ,download=False)
testset=torchvision.datasets.CIFAR10(root='/home/lshe/code_study/',train=False,transform=transforms.ToTensor(),download=False)
train_loader=torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True)
test_loader=torch.utils.data.DataLoader(testset,batch_size=batchsize)
train_loader1=torch.utils.data.DataLoader(trainset,batch_size=int(len(trainset)),shuffle=True)
test_loader1=torch.utils.data.DataLoader(testset,batch_size=int(len(testset)))



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

a=torch.randn(eqnum,difnum,steps,branch,device=cuda)
a.requires_grad_()


delta_a=torch.zeros_like(a)
for times in range(epoch):
    if((times+1)%20==0):
        learning_rate/=2

    with torch.no_grad():
        for x_train,y_train in train_loader1:
            x_train=x_train.to(cuda)

            y_train=y_train.to(cuda)
            yy = torch.zeros(y_train.size()[0], classes,device=cuda).scatter_(1, y_train.reshape(-1, 1), 1)
            y_train = yy
        # data_iter=iter(train_loader)
        # x_train,y_train=data_iter.next()
            n_x, f_x, w_x, h_x = x_train.size()

            xx = torch.zeros(n_x, eqnum, f_x, w_x, h_x,device=cuda)

            for i in range(f_x):

                for k in range(eqnum):
                    tt=x_train[::,i,::,::]
                    for j in range(steps):

                        p_x,p_y,p_xx,p_yy,p_xy=diff(tt)
                        xz = F.softsign(
                            a[k][0][j][0] + a[k][1][j][0] * tt + a[k][2][j][0] * (p_x * p_x + p_y * p_y) + a[k][3][j][
                                0] * (p_xx + p_yy) + a[k][4][j][0] *
                            (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][0] * (
                                        p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                        for jj in range(1, branch):
                            xz = xz + F.softsign(
                                a[k][0][j][jj] + a[k][1][j][jj] * tt + a[k][2][j][jj] * (p_x * p_x + p_y * p_y) +
                                a[k][3][j][jj] * (p_xx + p_yy) + a[k][4][j][jj] *
                                (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][jj] * (p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                        tt = xz + tt
                    xx[::,k,i,::,::]=tt
            # xx=xx.reshape(n_x,-1)
            del tt,p_x,p_y,p_xx,p_yy,p_xy,x_train
            # gc.collect()
            x=torch.zeros(n_x,xx.reshape(n_x,-1).size()[1]+1,device=cuda)
            kkk=xx.reshape(n_x,-1).size()[1]
            x[::,0:kkk]=xx.reshape(n_x,-1)
            x[::,kkk]=1
            W = torch.matmul(y_train.transpose(0, 1), x)
            ll=torch.matmul(x.transpose(0, 1), x) + lamb * n_x * torch.eye(x.size()[1], device=cuda)
            del x
            ll=torch.inverse(ll)
            W=torch.matmul(W,ll)
            del ll
            # W = torch.matmul(W, torch.inverse(
            #     torch.matmul(x.transpose(0, 1), x) + lamb * n_x * torch.eye(x.size()[1], device=cuda)))



    for x_train,y_train in train_loader:
        x_train=x_train.to(cuda)

        y_train=y_train.to(cuda)
        yy = torch.zeros(y_train.size()[0], classes,device=cuda).scatter_(1, y_train.reshape(-1, 1), 1)
        y_train = yy
    # data_iter=iter(train_loader)
    # x_train,y_train=data_iter.next()
        n_x, f_x, w_x, h_x = x_train.size()

        xx = torch.zeros(n_x, eqnum, f_x, w_x, h_x,device=cuda)

        for i in range(f_x):

            for k in range(eqnum):
                tt=x_train[::,i,::,::]
                for j in range(steps):

                    p_x,p_y,p_xx,p_yy,p_xy=diff(tt)
                    xz = F.softsign(
                        a[k][0][j][0] + a[k][1][j][0] * tt + a[k][2][j][0] * (p_x * p_x + p_y * p_y) + a[k][3][j][0] * (
                                    p_xx + p_yy) + a[k][4][j][0] *
                        (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][0] * (
                                    p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                    for jj in range(1, branch):
                        xz = xz + F.softsign(
                            a[k][0][j][jj] + a[k][1][j][jj] * tt + a[k][2][j][jj] * (p_x * p_x + p_y * p_y) +
                            a[k][3][j][jj] * (p_xx + p_yy) + a[k][4][j][jj] *
                            (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][jj] * (p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                    tt = xz + tt
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
        # with torch.no_grad():
        #     W = torch.matmul(y_train.transpose(0, 1), x)
        #     W = torch.matmul(W, torch.inverse(torch.matmul(x.transpose(0, 1), x) + lamb * n_x * torch.eye(x.size()[1],device=cuda)))

        # W=W.to(torch.float32)


        loss=torch.sum((torch.matmul(W,x.transpose(0,1))-y_train.transpose(0,1))*(torch.matmul(W,x.transpose(0,1))-y_train.transpose(0,1)))/n_x+lamb*torch.sum(W*W)

        loss.backward()
        with torch.no_grad():
            delta_a = delta_a * momentum + (1 - momentum) * a.grad
            a*=(1-decay)
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
            yy=torch.zeros(y_test.size()[0],classes,device=cuda).scatter_(1,y_test.reshape(-1,1),1)
            y_test=yy

            test_num, f_x, w_x, h_x = x_test.size()

            xx = torch.zeros(test_num, eqnum, f_x, w_x, h_x,device=cuda)

            for i in range(f_x):


                for k in range(eqnum):
                    tt = x_test[::, i, ::, ::]
                    # print(k)
                    for j in range(steps):
                        p_x, p_y, p_xx, p_yy, p_xy = diff(tt)
                        xz = F.softsign(
                            a[k][0][j][0] + a[k][1][j][0] * tt + a[k][2][j][0] * (p_x * p_x + p_y * p_y) + a[k][3][j][
                                0] * (
                                    p_xx + p_yy) + a[k][4][j][0] *
                            (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][0] * (
                                    p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                        for jj in range(1, branch):
                            xz = xz + F.softsign(
                                a[k][0][j][jj] + a[k][1][j][jj] * tt + a[k][2][j][jj] * (p_x * p_x + p_y * p_y) +
                                a[k][3][j][jj] * (p_xx + p_yy) + a[k][4][j][jj] *
                                (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][jj] * (
                                            p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                        tt = xz + tt
                    xx[::, k, i, ::, ::] = tt
            xx = xx.reshape(test_num, -1)#最后学到的feature
            xx_ = torch.zeros(test_num, xx.size()[1] + 1,device=cuda)
            xx_[::, 0:xx.size()[1]] = xx
            xx_[::, xx.size()[1]] = 1
            pred=torch.matmul(W,xx_.transpose(0,1))
            pred=torch.argmax(pred,dim=0)
            init=torch.argmax(y_test,dim=1)
            del xx_
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
                yy = torch.zeros(y_test.size()[0], classes,device=cuda).scatter_(1, y_test.reshape(-1, 1), 1)
                y_test = yy
                test_num, f_x, w_x, h_x = x_test.size()

                xx = torch.zeros(test_num, eqnum, f_x, w_x, h_x,device=cuda)

                for i in range(f_x):

                    for k in range(eqnum):
                        tt = x_test[::, i, ::, ::]
                        # print(k)
                        for j in range(steps):
                            p_x, p_y, p_xx, p_yy, p_xy = diff(tt)
                            xz = F.softsign(
                                a[k][0][j][0] + a[k][1][j][0] * tt + a[k][2][j][0] * (p_x * p_x + p_y * p_y) +
                                a[k][3][j][0] * (
                                        p_xx + p_yy) + a[k][4][j][0] *
                                (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][0] * (
                                        p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                            for jj in range(1, branch):
                                xz = xz + F.softsign(
                                    a[k][0][j][jj] + a[k][1][j][jj] * tt + a[k][2][j][jj] * (p_x * p_x + p_y * p_y) +
                                    a[k][3][j][jj] * (p_xx + p_yy) + a[k][4][j][jj] *
                                    (p_x * p_x * p_xx + 2 * p_x * p_y * p_xy + p_y * p_y * p_yy) + a[k][5][j][jj] * (
                                                p_xx * p_xx + 2 * p_xy * p_xy + p_yy * p_yy))
                            tt = xz + tt
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

