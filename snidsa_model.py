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
