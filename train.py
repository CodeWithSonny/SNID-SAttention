
#!/usr/bin/python
# -*- coding: utf-8 -*-
from data_utils import *
import os.path
import numpy as np
import tensorflow as tf
import argparse
import sys
from tqdm import tqdm
from snidsa_model import SNIDSA

class Config(object):
    """Configuration of model"""
    num_layers = 1
    batch_size = 32
    embedding_dim = 32
    hidden_dim = 64
    num_epochs = 200
    valid_freq = 5
    patience = int(10/valid_freq) + 1
    model = 'snidsa'
    gpu_no = '0'
    data_name = 'data/hc-exp'
    learning_rate = 0.001
    dropout = 1
    random_seed = 1402

class Input(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        self.num_nodes = config.num_nodes
        self.inputs, self.targets, self.seq_lenghth = batch_generator(data, self.batch_size)
        self.batch_num = len(self.inputs)
        self.cur_batch = 0

    def next_batch(self):
        x = self.inputs[self.cur_batch]
        y = self.targets[self.cur_batch]
        sl = self.seq_lenghth[self.cur_batch]
        self.cur_batch = (self.cur_batch +1) % self.batch_num
        batch_size = x.shape[0]
        num_steps = x.shape[1]
        return x, y, sl, batch_size, num_steps

def rank_eval(pred, true_labels, sl):
    mrr = 0
    ac1 = 0
    ac5 = 0
    ac10 = 0
    ac50 = 0
    ac100 = 0
    num_nodes = pred.shape[2]
    for i in range(len(sl)):
        length = sl[i]
        for j in range(length):
            y_pos = true_labels[i][j]
            predY = pred[i][j][y_pos]
            rank = 1.
            for k in range(num_nodes):
                if pred[i][j][k]> predY:
                    rank += 1.
            if rank <= 1:
                ac1 += 1./float(length)
            if rank <= 5:
                ac5 += 1./float(length)
            if rank <= 10:
                ac10 += 1./float(length)
            if rank <= 50:
                ac50 += 1./float(length)
            if rank <= 100:
                ac100 += 1./float(length)
            mrr += (1./rank)/float(length)
    return mrr, ac1, ac5, ac10, ac50, ac100

def args_setting(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lr", type=float, help="learning rate")
    parser.add_argument("-x", "--xdim", type=int, help="embedding dimension")
    parser.add_argument("-e", "--hdim", type=int, help="hidden dimension")