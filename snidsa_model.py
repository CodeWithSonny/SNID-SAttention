import tensorflow as tf
from snidsa_cell import *

class SNIDSA(object):
    def __init__(self, config, A, is_training=True):

        self.num_nodes = config.num_nodes
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        # self.model = config.model

        self.learning_rate = config.learning_rate
        self.dropout = config.dropout

        self._A = tf.constant(A, dtype=tf.float32, name="adjacency_matrix")

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [self.num_nodes,
                    self.embedding_dim], dtype=tf.float32)

        self.placeholders()
        self.loss_mask()
        self.graph_information()
        self.recurrent_layer()
        self.cost()
        self.optimize()

    def placeholders(self):
        self.batch_size = tf.placeholder(tf.int32, None)
        self._inputs = tf.placeholder(tf.int32, [None, None]) # [batch_size, num_steps]
        self._targets = tf.placeholder(tf.int32, [None, None])
        self._seqlen = tf.placeholder(tf.int32, [None])
        self.num_steps = tf.placeholder(tf.int32, None)

    def loss_mask(self):
        self._target_mask = tf.sequence_mask(self._se