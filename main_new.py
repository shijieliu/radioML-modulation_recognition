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

class conv_layer(nn.Module):
    def __init__(self):
        super(conv_layer,self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(1,64,(1,1)))
        #layers.append(nn.BatchNorm2d(64))
        layers.append(identity_block(64))
        #layers.append(nn.Conv2d(64,1,1))
        self.layers = layers

    def forward(self,input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class identity_block(nn.Module):
    def __init__(self,channel):
        super(identity_block,self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(channel,2*channel,(3,3),padding = (1,1)))
        layers.append(nn.BatchNorm2d(2 * channel))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(channel * 2,channel * 2, (3,3), padding = (1,1)))
        layers.append(nn.BatchNorm2d(2 * channel))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(channel * 2,channel, (1,1)))
        layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.ReLU(inplace = True))
        self.layers = layers

    def forward(self,input):
        x = input
        for layer in self.layers:
            x = layer(x)
        out = x + input
        return out


class radio_cnn(nn.Module):
    def __init__(self,batch = 64):
        super(radio_cnn,self).__init__()
        self.batch = batch
        self.conv = conv_layer()
        #self.conv = nn.DataParallel(self.conv)
        #self.linear = nn.Sequential(
        #        nn.Linear(256,256),
        #        nn.ReLU(inplace = True)
        #        )
        linearLayer = nn.ModuleList()
        linearLayer.append(nn.Linear(16384,256))
        #linearLayer.append(nn.Dropout(0.5, inplace = True))
        linearLayer.append(nn.Linear(256,11))
        #linearLayer.append(nn.Dropout(0.5, inplace = True))
        self.linearLayer = linearLayer

    def forward(self,input):
        #variable
        #output = self.conv(input)
        #output = output.view(self.batch,-1)
        #output = self.linear(output)
        output = self.conv(input)
        output = output.view(self.batch,-1)
        for layer in self.linearLayer:
            output = layer(output)
        #print torch.sum(output[0,:].data)
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

def build_conv(batch):
    return radio_cnn(batch)

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

def main(FLAGS):
    train_data,test_data = init_data(FLAGS)
    val_loader = data.DataLoader(test_data, FLAGS.batch,num_workers = 8, \
            shuffle = True, pin_memory = True, drop_last =True)
    data_loader = data.DataLoader(train_data, FLAGS.batch, num_workers = 8,\
            shuffle = True, pin_memory = True, drop_last = True)
    conv_net = build_conv(FLAGS.batch)

    if FLAGS.weights:
        conv_net.load_state_dict(torch.load(FLAGS.weights),strict =False)
    else:
        for m in conv_net.modules():
            if isinstance(m,nn.Conv2d):
                init.xavier_normal(m.weight.data)
                init.constant(m.bias.data,0)
            if isinstance(m,nn.Linear):
                init.xavier_normal(m.weight.data)
                init.constant(m.bias.data,0)

    #for k,v in conv_net.state_dict().items():
    #     print k,v

    #net = nn.DataParallel(conv_net)
    net = conv_net.cuda()
    #cudnn.benchmark = True

    #optimizer = optim.Adam(net.parameters(), lr = FLAGS.lr)
    #optimizer = optim.SGD(net.parameters(), lr = FLAGS.lr) \
    #        momentum = FLAGS.momentum)
    #        momentum = FLAGS.momentum, weight_decay = FLAGS.weight_decay)
    optimizer = optim.Adagrad(net.parameters(), lr = FLAGS.lr, weight_decay = FLAGS.weight_decay)
    criterion = build_loss(FLAGS)

    vis = visdom.Visdom()
    lot = vis.line( \
            X = torch.zeros((1,)).cpu(), \
            Y = torch.zeros((1,)).cpu(), \
            opts = dict( \
                xlabel = "iteration",
                ylabel = "loss",
                title = 'loss',
                legend = ['train loss']
                )
            )
    lot2 = vis.line( \
            X = torch.zeros((1,)).cpu(), \
            Y = torch.zeros((1,2)).cpu(), \
            opts = dict( \
                xlabel = "iteration",
                ylabel = "accuracy",
                title = 'accuracy',
                legend = ['train accuracy', 'validation accuracy']
                )
            )

    


    iteration = 0
    lr = FLAGS.lr
    #for epoch in range(1,FLAGS.epochs+1):
    epoch = 1
    while True:
        #if epoch %2 == 0:
        #    lr *= 10
        #    adjust_learning_rate(optimizer,lr)
        #    print lr,iteration
        net.train()

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
                #print torch.mean(out.data.cpu())
                #print torch.autograd.grad( \
                #        loss, \
                #        out, \
                        #grad_outputs = torch.ones(loss.data.shape).cuda(), \
                        #create_graph = True, \
                        #retain_graph = True \
                #        
        epoch += 1
        
        net.eval()
        train_accu = 0
        train_total = 0
        for index, batchdata in enumerate(data_loader):
            x = Variable(batchdata[0].cuda())
            label = batchdata[1]
            out = net(x)
            predict,indices = torch.max(out,1)
            indices = indices.data.cpu().unsqueeze(dim = 1)
            train_accu += (indices.eq(label)).sum()
            train_total += indices.shape[0]
        print "train acurrent accuracy = %f" % (train_accu/float(train_total))


        val_accu = 0
        val_total = 0
        for index, batchdata in enumerate(val_loader):
            x = Variable(batchdata[0].cuda())
            label = batchdata[1]
            out = net(x)
            predict,indices = torch.max(out,1)
            indices = indices.data.cpu().unsqueeze(dim = 1)
            val_accu += (indices.eq(label)).sum()
            val_total += indices.shape[0]
        print "test current accuracy = %f" % (val_accu/float(val_total))

        vis.line( \
                    X = torch.ones((1,2)).cpu() * iteration, \
                    Y = torch.Tensor([train_accu/float(train_total), val_accu/float(val_total)]).unsqueeze(0).cpu(), \
                    win = lot2, \
                    update = 'append' \
                    )


        if epoch % 10 == 0:
            save_dir = os.path.join(FLAGS.param,'.'.join(["param",str(epoch),'.pth']))
            torch.save(net.state_dict(),save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type = str, default = "/home/v-shliu/code/radioML-modulation_recognition/RML2016.10a_dict.dat")
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--momentum",type = float, default = 0.9)
    parser.add_argument("--weight_decay",type = float, default = 1e-2)
    parser.add_argument("--batch",type = int, default = 1024)
    parser.add_argument("--epochs", type = int, default =100)
    parser.add_argument("--phase", type = str, default = "test")
    parser.add_argument("--param", type = str)
    parser.add_argument("--weights",type = str)
    parser.add_argument("--confident_penalty", type = bool, default = False)
    parser.add_argument("--use_gpu", type = str, default = "0")

    FLAGS = parser.parse_args()
    print FLAGS

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.use_gpu

    if FLAGS.param == None:
        raise NotImplementedError()
    if not os.path.exists(FLAGS.param):
        os.mkdir(FLAGS.param)

    main(FLAGS)
