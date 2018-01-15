#!/usr/bin/env python
#encoding=utf-8

import os
from argparse import ArgumentParser
import numpy as np
from utils import init_data,ConvModel

def main(FLAGS):
    train_data, test_data = init_data(FLAGS)
    convmodel = ConvModel(train_data, test_data, FLAGS)
    convmodel.compile()
    convmodel.train()





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/v-shliu/code/radioML-modulation_recognition/RML2016.10a_dict.dat")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum",type = float, default=0.9)
    parser.add_argument("--weight_decay",type=float, default=0.0)
    parser.add_argument("--weights",type=str, default=None)
    parser.add_argument("--batch",type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--confident_penalty", type = bool, default = False)
    parser.add_argument("--use_gpu", type = str, default = "0")

    FLAGS = parser.parse_args()
    print FLAGS

    main(FLAGS)
