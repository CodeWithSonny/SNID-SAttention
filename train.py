
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