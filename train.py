
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
    parser.add_argument("-d", "--data", help="data name")
    parser.add_argument("-g", "--gpu", help="gpu id")
    parser.add_argument("-b", "--bs", type=int, help="batch size")
    parser.add_argument("-f", "--freq", type=int, help="validation frequency")
    parser.add_argument("-n", "--nepoch", type=int, help="number of training epochs")
    args = parser.parse_args()
    if args.lr:
        config.learning_rate = args.lr
    if args.xdim:
        config.embedding_dim = args.xdim
    if args.hdim:
        config.hidden_dim = args.hdim
    if args.bs:
        config.batch_size = args.bs
    if args.data:
        config.data_name = args.data
    if args.gpu:
        config.gpu_no = args.gpu
    if args.freq:
        config.valid_freq = args.freq
        config.patience = int(10/config.valid_freq) + 1
    if args.nepoch:
        config.num_epochs = args.nepoch
    return config

def train(argv):
    config = Config()
    config = args_setting(config)

    data_name = config.data_name
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_no
    num_epochs = config.num_epochs

    # data load
    train_data, valid_data, test_data, nodes, node_to_id = \
        read_raw_data(data_name + '-cascades')
    config.num_nodes = len(nodes)
    train_size = train_data[2]
    valid_size = valid_data[2]
    test_size = test_data[2]
    print (train_size, valid_size, test_size)
    A = read_graph(data_name + '-graph', node_to_id)
    input_train = Input(config, train_data)
    input_valid = Input(config, valid_data)
    input_test = Input(config, test_data)

    # Model create
    model = SNIDSA(config, A)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)
    tf.set_random_seed(config.random_seed)

    # Parameter Initialization
    sess.run(tf.global_variables_initializer())

    # Record test results at best validation epoch with early stopping
    max_logits = float('inf')
    stop_count = 0
    best_mrr = 0
    best_ac1 = 0
    best_ac5 = 0
    best_ac10 = 0
    best_ac50 = 0
    best_ac100 = 0
    best_valid_epoch = 0

    # Print Training information
    train_info = "Data: {0}, Model: {1}, GPU Num: {2}, Learning Rate: {3:.3f}, Embedding Size: {4}, Hidden Size: {5}, Batch Size: {6}"
    print(train_info.format(config.data_name, config.model, config.gpu_no, config.learning_rate, config.embedding_dim, config.hidden_dim, config.batch_size))
    print('Start training...')

    # Training Process
    for epoch in range(num_epochs):
        epoch_logits = 0
        valid_logits = 0
        # test_logits = 0
        valid_mrr = 0
        valid_ac1 = 0
        valid_ac5 = 0
        valid_ac10 = 0
        valid_ac50 = 0
        valid_ac100 = 0
        test_mrr = 0
        test_ac1 = 0
        test_ac5 = 0
        test_ac10 = 0
        test_ac50 = 0
        test_ac100 = 0

        msg = 'Epoch ' + str(epoch+1) + '/' + str(num_epochs) + ' (Train)'
        for i in tqdm(range(input_train.batch_num), desc=msg):