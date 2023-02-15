
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
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        X = inputs[0]
        A = inputs[1]
        feat_in = self._feat_in
        feat_out = self._num_units
        num_nodes = self._num_nodes
        feat_in_matrix = self._feat_in_matrix

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Attentions"):
                struc, x = struc_atten(X, feat_in_matrix, A, feat_in)
            with tf.variable_scope("c_inputs"):
                xs_c = linear([x, struc], feat_out, False)
            with tf.variable_scope("h_inputs"):
                xs_h = linear([x, struc], feat_out, False)
            with tf.variable_scope("Gate"):
                concat = tf.sigmoid(
                    linear([X, struc], 2 * feat_out, True))
                if tf.__version__ == "0.12.1":
                    f, r = tf.split(1, 2, concat)
                else:
                    f, r = tf.split(axis=1, num_or_size_splits=2, value=concat)

            c = f * state + (1 - f) * xs_c

            # highway connection
            h = r * self._activation(c) + (1 - r) * xs_h

        return h, c


def struc_atten(X, feat_in_matrix, A, feat_out):
    batch_size = tf.shape(X)[0]
    num_nodes = tf.shape(feat_in_matrix)[0]
    with tf.variable_scope("linear_transf"):
        linear_transf_X = linear([X], feat_out, False)
        tf.get_variable_scope().reuse_variables()
        linear_transf_G = linear([feat_in_matrix], feat_out, False)
    with tf.variable_scope("strcuture_attention"):
        Wa = tf.get_variable("Wa", [2*feat_out, 1], dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # Repeat feature vectors of input: [[1], [2]] becomes [[1], [1], [2], [2]]
        repeated = tf.reshape(tf.tile(linear_transf_X, (1, num_nodes)), (batch_size * num_nodes, feat_out))  # (BN x F')
        # Tile feature vectors of full graph: [[1], [2]] becomes [[1], [2], [1], [2]]
        tiled = tf.tile(linear_transf_G, (batch_size, 1))  # (BN x F')
        # Build combinations