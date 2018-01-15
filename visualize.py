#!/usr/bin/env python
#encoding=utf-8

import visdom
import numpy as np
import torch

class display(object):
    def __init__(self):
        self.vis = visdom.Visdom()

    def draw(self, X, Y):
        self.vis.line(X=X, \
                Y=Y, \
                win=self.lot, \
                update='append')


class display_lr(display):
    def __init__(self):
        super(display_lr, self).__init__()
        self.lot = self.vis.line( \
                X=torch.zeros((1,)).cpu(), \
                Y=torch.zeros((1,)).cpu(), \
                opts=dict( \
                    xlabel="iteration", \
                    ylabel="learning rate", \
                    title="learning rate", \
                    legend=["learning rate"]))

    def cal(self, optims):
        lr = 0.0
        for param_group in optims.param_groups:
            lr += param_group['lr']
        return lr / len(optims.param_groups)


class display_loss(display):
    def __init__(self):
        super(display_loss, self).__init__()
        self.lot = self.vis.line( \
                X=torch.zeros((1,)).cpu(), \
                Y=torch.zeros((1,)).cpu(), \
                opts=dict( \
                    xlabel="iteration", \
                    ylabel="loss", \
                    title="loss", \
                    legend=["train loss"]))


class display_accuracy(display):
    def __init__(self):
        super(display_accuracy, self).__init__()
        self.lot = self.vis.line( \
                X=torch.zeros((1,)).cpu(), \
                Y=torch.zeros((1,2)).cpu(), \
                opts=dict( \
                    xlabel="iteration", \
                    ylabel="accuracy", \
                    title="accuracy", \
                    legend=['train accuracy', 'validate accuracy']))
