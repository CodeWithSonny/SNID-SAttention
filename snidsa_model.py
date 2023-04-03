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
        self._target_mask = tf.sequence_mask(self._seqlen, dtype=tf.float32)

    def graph_information(self):
        _neighbors = tf.nn.embedding_lookup(self._A, self._inputs)
        return _neighbors

    def input_embedding(self):
        _inputs = tf.nn.embedding_lookup(self.embedding, self._inputs)
        return _inputs

    def recurrent_layer(self):
        def creat_cell():
            cell = SnidsaCell(self.hidden_dim, self.embedding)
            if self.dropout < 1:
                return tf.contrib.rnn.DropoutWrapper(cell,
                    output_keep_prob=self.dropout)
            else:
                return cell

        cells = [creat_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        emb_inputs = self.input_embedding()
        _neighbors = self.graph_information()
        _outputs, _ = tf.nn.dynamic_rnn(cell=cell,
            inputs=(emb_inputs,_neighbors), sequence_length=self._seqlen, dtype=tf.float32)

        output = tf.reshape(tf.concat(_outputs, 1), [-1, self.hidden_dim])
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_dim, self.num_nodes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.num_nodes], dtype=tf.float32)
        self.flat_logi