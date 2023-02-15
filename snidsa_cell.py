
#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class SnidsaCell(RNNCell):
    """ Recurrent Unit Cell for SNIDSA."""

    def __init__(self, num_units, feat_in_matrix, activation=None, reuse=None):
        self._num_units = num_units
        self._num_nodes = int(feat_in_matrix.shape[0])
        self._feat_in_matrix = feat_in_matrix
        self._feat_in = int(feat_in_matrix.shape[1])
        self._activation = activation or tf.tanh

    @property