#!/usr/bin/env python
#encoding=utf-8

import os
from argparse import ArgumentParser
import cPickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.optim as optim
import visdom
from itertools import product
import torch.nn.functional as F
import numpy as np

class fm_data(data.Dataset):
    def __init__(self,x,y):
        super(fm_data,self).__init__()
        #self.x = x[:128]
        self.x = x
        #self.y = y[:128]
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        return torch.from_numpy(self.x[index]).unsqueeze(dim = 0), \
                torch.LongTensor([self.y[index]])

class multiloss(nn.Module):
    def __init__(self):
        super(multiloss,self).__init__()
        #self.softmax = nn.LogSoftmax()
        #self.loss = nn.NLLLoss()
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, target, label):
        #都是variable
        #target = self.softmax(target)
        label = label[:,0].clone()
        #return self.loss(target,label)
        return self.crossentropy(target,label)


class radio_cnn(nn.Module):
    def __init__(self,batch = 64):
        super(radio_cnn,self).__init__()
        self.batch = batch
        self.conv = nn.ModuleList([
                nn.Conv2d(1,256,(1,3),padding = (0,2)),
                nn.ReLU(inplace = True),
                #nn.Dropout(p = 0.5,inplace = False),
                nn.Conv2d(256,80,(2,3),padding = (0,2)),
                nn.ReLU(inplace = True),
                #nn.Dropout(p = 0.5, inplace = False)
                ])
        self.linear = nn.Sequential(
                nn.Linear(10560,256),
                nn.ReLU(inplace = True)
                )
        #self.linear = nn.Sequential(
        #        nn.Linear(256,256),
        #        nn.ReLU(inplace = True)
        #        )
        self.output_layer = nn.Linear(256,11)


    def forward(self,input):
        #variable
        #output = self.conv(input)
        #output = output.view(self.batch,-1)
        #output = self.linear(output)
        output = input
        for i in self.conv:
            output = i(output)
        output = output.view(self.batch,-1)
        output = self.linear(output)
        output = self.output_layer(output)
        #print torch.sum(output[0,:].data)
        #raise
        return output

def init_data(FLAGS):
    Xd = cPickle.load(open(FLAGS.data,"rb"))
    #small_data = {}
    #for index,(k,v) in enumerate(Xd.keys()):
    #    if index > 100:
    #        break
    #    small_data[k] = v
    #f = open('small_data.pkl','wb')
    #cPickle.dump(small_data,f)
    #f.close()

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j],Xd.keys())))),[1,0])
    #print Xd
    #print snrs
    #print mods
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            elem = (mod,snr)
            X.append(Xd[elem])
            for i in range(Xd[elem].shape[0]):
                lbl.append(elem)
    X = np.vstack(X)
    np.random.seed(2017)
    n_example = X.shape[0]
    n_train = n_example * 7 // 10
    train_idx = np.random.choice(range(0,n_example), size = n_train, replace = False)
    test_idx = list(set(range(0,n_example)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = map(lambda x : mods.index(lbl[x][0]),train_idx)
    Y_test = map(lambda x : mods.index(lbl[x][0]),test_idx)
    #print X_train, X_train.shape
    #print Y_train, Y_train.shape
    train_data = fm_data(X_train,Y_train)
    test_data = fm_data(X_test,Y_test)
    return train_data,test_data


def build_conv(batch):
    return radio_cnn(batch)

def build_loss(FLAGS):
    return multiloss()

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
    if isinstance(m,nn.Linear):
        init.normal(m.weight.data)

def adjust_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(FLAGS):
    train_data,test_data = init_data(FLAGS)
    data_loader = data.DataLoader(train_data, FLAGS.batch, num_workers = 8,\
            shuffle = True, pin_memory = True, drop_last = True)
    conv_net = build_conv(FLAGS.batch)

    for m in conv_net.modules():
        if isinstance(m,nn.Conv2d):
            init.xavier_normal(m.weight.data)
            init.normal(m.bias.data,std = 0.1)
        if isinstance(m,nn.Linear):
            init.normal(m.weight.data,std = 0.1)
            init.normal(m.bias.data,std = 0.1)

    #for k,v in conv_net.state_dict().items():
    #     print k,v

    #net = nn.DataParallel(conv_net)
    net = conv_net.cuda()
    cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr = FLAGS.lr)
    #optimizer = optim.SGD(net.parameters(), lr = FLAGS.lr, \
    #        momentum = FLAGS.momentum)
            #momentum = FLAGS.momentum, weight_decay = FLAGS.weight_decay)
    #optimizer = optim.Adagrad(net.parameters(), lr = FLAGS.lr)
    criterion = build_loss(FLAGS)

    vis = visdom.Visdom()
    lot = vis.line( \
            X = torch.zeros((1,)).cpu(), \
            Y = torch.zeros((1,)).cpu(), \
            opts = dict( \
                xlabel = "iteration",
                ylabel = "loss",
                title = 'training loss',
                legend = ['loss']
                )
            )

    iteration = 0
    lr = FLAGS.lr
    print len(data_loader)
    for epoch in range(1,FLAGS.epochs+1):
        #if epoch %2 == 0:
        #    lr *= 10
        #    adjust_learning_rate(optimizer,lr)
        #    print lr,iteration

        for index, batchdata in enumerate(data_loader):
            iteration += 1
            x = Variable(batchdata[0].cuda(), requires_grad = True)
            label = Variable(batchdata[1].cuda(), requires_grad = False)
            out = net(x)
            loss = criterion(out,label)

            #print torch.mean( \
            #        torch.autograd.grad( \
            #            loss, \
            #            x, \
            #            grad_outputs = torch.ones(loss.data.shape).cuda(), \
            #            create_graph = True, \
            #            retain_graph = True
            #            )
            #        )
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            vis.line( \
                    X = torch.ones((1,)).cpu() * iteration, \
                    Y = torch.Tensor([loss.data[0]]).cpu(), \
                    win = lot, \
                    update = 'append' \
                    )
            if iteration % 20 == 0:
                pass
                #print torch.mean(out.data.cpu())
                #print torch.autograd.grad( \
                #        loss, \
                #        out, \
                        #grad_outputs = torch.ones(loss.data.shape).cuda(), \
                        #create_graph = True, \
                        #retain_graph = True \
                #        )

        if epoch % 10 == 0:
            save_dir = os.path.join('./param','.'.join(["param",str(epoch),'.pth']))
            torch.save(net.state_dict(),save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type = str, default = "/home/shijie/code/convolution_fm/RML2016.10a_dict.dat")
    parser.add_argument("--lr", type = float, default = 1e-2)
    parser.add_argument("--momentum",type = float, default = 0.9)
    parser.add_argument("--weight_decay",type = float, default = 5e-4)
    parser.add_argument("--batch",type = int, default = 1024)
    parser.add_argument("--epochs", type = int, default =100)

    FLAGS = parser.parse_args()
    print FLAGS

    main(FLAGS)
