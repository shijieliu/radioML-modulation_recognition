#!/usr/bin/env python
#encoding=utf-8

import os
import torch
from torch.autograd import Variable
import cPickle
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.optim as optim
import visdom
from itertools import product
import torch.nn.functional as F
from visualize import display_lr, display_loss, display_accuracy
from model import conv_layer
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


class output_penalty(nn.Module):
    def __init__(self, epsilon):
        super(output_penalty,self).__init__()
        self.eps = epsilon
    
    def forward(self,target,label):
        '''
        label : (batchsize,)
        target : (batchsize, 11)
        '''
        label = torch.unsqueeze(label,1)
        label_onehot = torch.zeros_like(target).float()
        label_onehot.scatter_(1,label,1).cuda()
        target  = target.float()
        softmax = F.softmax(target, dim = 1)
        logsoftmax = F.log_softmax(target,dim = 1)
        entropy = label_onehot * softmax * logsoftmax
        return self.eps  * entropy.sum()

class multiloss(nn.Module):
    def __init__(self, flag_penalty):
        super(multiloss,self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()
        self.flag_penalty = flag_penalty
        if flag_penalty:
            self.penalty = output_penalty(1e-3)

    def forward(self, target, label):
        #都是variable
        label = label[:,0].clone()
        if self.flag_penalty:
            return self.crossentropy(target, label) + self.penalty(target, label)
        return self.crossentropy(target,label)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)





class old_conv_model(nn.Module):
    def __init__(self, batch=1024, dr=0.5):
        super(old_conv_model, self).__init__()
        self.batch = batch
        self.dr = dr
        self.conv = nn.ModuleList()
        self.linearLayer = nn.ModuleList()
        self.build_conv()
        self.build_linear()

    def build_conv(self):
        self.conv.append(nn.Conv2d(1, 256, (1,3), padding=(0,2)))
        self.conv.append(nn.ReLU(inplace=True))
        self.conv.append(nn.Dropout(self.dr, inplace=True))
        self.conv.append(nn.Conv2d(256, 80, (2,3), padding=(0,2)))
        self.conv.append(nn.ReLU(inplace=True))
        self.conv.append(nn.Dropout(self.dr, inplace=True))

    def build_linear(self):
        self.linearLayer.append(nn.Linear(10560, 256))
        self.linearLayer.append(nn.ReLU(inplace=True))
        self.linearLayer.append(nn.Dropout(self.dr, inplace=True))

        self.linearLayer.append(nn.Linear(256, 11))
        self.linearLayer.append(nn.Softmax(dim=1))

    def forward(self, x):
        for i in self.conv:
            x = i(x)
        x = x.view(self.batch, -1)
        for i in self.linearLayer:
            x = i(x)
        return x

class baseModel(object):
    def __init__(self, trainset, valset, FLAGS):
        self.trainset = trainset
        self.valset = valset
        self.FLAGS = FLAGS
        self.model = None

        self.dis_lr = display_lr()
        self.dis_loss = display_loss()
        self.dis_accu = display_accuracy()

    def init_weights(self):
        if self.model is None:
            raise ValueError("model value")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weights.data)
                init.constant(m.bias.data, 0)
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weights.data)
                init.constant(m.bias.data, 0)

    def load_weights(self, weights_dir):
        if self.model is None:
            raise ValueError
        self.model.load_state_dict(torch.load(weights_dir), strict = True)

    def prepare_optim(self, optims=None):
        if optims is None:
            self.optim = optim.SGD(self.model.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)
        elif optims == "adam_ori":
            self.optim = optim.Adam(self.model.parameters())
        elif optims == "adam":
            self.optim = optim.Adam(self.model.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)
        else:
            raise NotImplementedError

    def prepare_dataloader(self, dataset):
        return data.DataLoader(dataset, \
                self.FLAGS.batch, \
                num_workers=8, \
                shuffle=True, \
                pin_memory=True, \
                drop_last=True)

    def train(self):
        
        iteration = 0
        epoch = 0
        while True:
            self.model.train()
            for index, batchdata in enumerate(self.trainloader):
                iteration += 1
                x = Variable(batchdata[0].cuda(), requires_grad=True)
                label = Variable(batchdata[1].cuda(), requires_grad=False)
                out = self.model(x)
                loss = self.criterion(out, label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.dis_loss.draw(X=torch.ones((1,)).cpu() * iteration, \
                        Y=torch.Tensor([loss.data[0]]).cpu())
                average_lr = self.dis_lr.cal(self.optim)
                self.dis_lr.draw(X=torch.ones((1,)).cpu() * iteration, \
                        Y=torch.ones((1,)).cpu() * average_lr )
            epoch += 1
            train_accu = self.eval(flag='train')
            val_accu = self.eval(flag='validate')
            self.dis_accu.draw(X=torch.ones((1,2)).cpu() * iteration, \
                    Y=torch.Tensor([ train_accu, val_accu]).unsqueeze(0).cpu())

    def eval(self, flag):
        self.model.eval()
        
        assert flag in ['train', 'validate']
        dataloader = None
        if flag == 'train':
            dataloader = self.trainloader
        if flag == 'validate':
            dataloader = self.valloader

        train_accu = 0.0
        train_total = 0.0
        for index, batchdata in enumerate(dataloader):
            x = Variable(batchdata[0].cuda())
            label = batchdata[1]
            out = self.model(x)
            predict,indices = torch.max(out,1)
            indices = indices.data.cpu().unsqueeze(dim = 1)
            train_accu += (indices.eq(label)).sum()
            train_total += indices.shape[0]
        return train_accu / train_total

    def save(self, epoch):
        save_dir = os.path.join(self.FLAGS.param,'.'.join(["param",str(epoch),'.pth']))
        torch.save(self.model.state_dict(),save_dir)


class ConvModel(baseModel):
    def __init__(self, trainset, valset, FLAGS):
        super(ConvModel, self).__init__(trainset, valset, FLAGS)

    def compile(self, optim=None):
        self.trainloader = self.prepare_dataloader(self.trainset)
        self.valloader = self.prepare_dataloader(self.valset)

        self.model = old_conv_model()
        self.model.cuda()
        self.prepare_optim(self.FLAGS.optim)
        self.criterion = build_loss(self.FLAGS)


class ResModel(baseModel):
    def __init__(self, trainset, valset, FLAGS):
        super(ResModel, self).__init__(trainset, valset, FLAGS)

    def compile(self, optim=None):
        self.trainloader = self.prepare_dataloader(self.trainset)
        self.valloader = self.prepare_dataloader(self.valset)

        self.model = res_cnn()
        self.model.cuda()
        self.prepare_optim(self.FLAGS.optim)
        self.criterion = build_loss(self.FLAGS)

class res_cnn(nn.Module):

    def __init__(self, batch=1024, parallel=False, dropout=False):
        super(res_cnn, self).__init__()
        self.batch = batch
        self.conv = conv_layer()
        if parallel:
            self.conv = nn.DataParallel(self.conv)
        # self.linear = nn.Sequential(
        #        nn.Linear(256,256),
        #        nn.ReLU(inplace = True)
        #        )
        linearLayer = nn.ModuleList()
        linearLayer.append(nn.Linear(16384, 256))
        if dropout:
            linearLayer.append(nn.Dropout(0.5, inplace = True))
        linearLayer.append(nn.Linear(256, 11))
        if dropout:
            linearLayer.append(nn.Dropout(0.5, inplace = True))
        self.linearLayer = linearLayer

    def forward(self, input):
        # variable
        # output = self.conv(input)
        # output = output.view(self.batch,-1)
        # output = self.linear(output)
        print input.data.shape
        output = self.conv(input)
        output = output.view(self.batch, -1)
        for layer in self.linearLayer:
            output = layer(output)
        # print torch.sum(output[0,:].data)
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
    show_data(Y_train)
    show_data(Y_test)
    #print X_train, X_train.shape
    #print Y_train, Y_train.shape
    train_data = fm_data(X_train,Y_train)
    test_data = fm_data(X_test,Y_test)
    return train_data,test_data

def show_data(Y):
    unique, counts = np.unique(Y,return_counts = True)
    print dict(zip(unique, counts))


def build_loss(FLAGS):
    return multiloss(FLAGS.confident_penalty)

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
    if isinstance(m,nn.Linear):
        init.normal(m.weight.data)

def adjust_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr