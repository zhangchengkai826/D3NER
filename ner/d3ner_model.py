# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\ner\d3ner_model.py
# Compiled at: 2018-07-26 00:36:54
# Size of source mod 2**32: 10279 bytes
import numpy as np, tensorflow as tf
from keras.layers import Input
from ner.data_utils import pad_sequences
from constants import ALL_LABELS, ENTITY_TYPES
from train.constants import NCHARS, CHAR_DIM, NPOS, POS_DIM, CHAR_HIDDEN_SIZE, UNK, N_HIDDEN_SIZE
seed = 13
np.random.seed(seed)

class BiLSTMCRFCore:

    def __init__(self):
        self.embeddings = None
        self.npos = NPOS
        self.pos_dim = POS_DIM
        self.nchars = NCHARS
        self.char_dim = CHAR_DIM
        self.char_hidden_size = CHAR_HIDDEN_SIZE
        self.n_hidden_size = N_HIDDEN_SIZE
        self.nen_label_size = len(ENTITY_TYPES)
        self.all_labels = ALL_LABELS
        self.unk = UNK
        self.num_of_class = len(self.all_labels)

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.labels = Input(name='y_true', shape=[None], dtype='int32')
        self.label_lens = Input(name='lb_lens', batch_shape=[None], dtype='int32')
        self.nen_labels = Input(name='y_nen', shape=[None, self.nen_label_size], dtype='float32')
        self.word_ids = Input(name='word_ids', shape=[None], dtype='int32')
        self.char_ids = Input(name='char_ids', shape=[None, None], dtype='int32')
        self.word_lengths = Input(name='word_lengths', shape=[None], dtype='int32')
        self.dropout_op = tf.placeholder(dtype=(tf.float32), shape=[], name='dropout_op')
        self.dropout_lstm = tf.placeholder(dtype=(tf.float32), shape=[], name='dropout_lstm')
        self.word_pos_ids = Input(name='word_pos', shape=[None], dtype='int32')
        self.is_training = tf.placeholder((tf.bool), name='phase')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope('words'):
            _word_embeddings = tf.Variable((self.embeddings), name='_word_embeddings', dtype=(tf.float32), trainable=False)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, (self.word_ids), name='word_embeddings')
        with tf.variable_scope('pos'):
            _pos_embeddings = tf.get_variable(name='_pos_embeddings', dtype=(tf.float32), shape=[
             self.npos, self.pos_dim],
              initializer=(tf.contrib.layers.xavier_initializer()))
            pos_embeddings = tf.nn.embedding_lookup(_pos_embeddings, (self.word_pos_ids), name='pos_embeddings')
            sh = tf.shape(pos_embeddings)
            pos_embeddings = tf.reshape(pos_embeddings, shape=[-1, sh[(-2)], self.pos_dim])
        with tf.variable_scope('chars'):
            _char_embeddings = tf.get_variable(name='_char_embeddings', dtype=(tf.float32), shape=[
             self.nchars, self.char_dim],
              initializer=(tf.contrib.layers.xavier_initializer()))
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, (self.char_ids), name='char_embeddings')
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[(-2)], self.char_dim])
            word_lengths = tf.reshape((self.word_lengths), shape=[-1])
            cell_fw = tf.contrib.rnn.LSTMCell((self.char_hidden_size), initializer=(tf.contrib.layers.xavier_initializer()))
            cell_bw = tf.contrib.rnn.LSTMCell((self.char_hidden_size), initializer=(tf.contrib.layers.xavier_initializer()))
            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
              char_embeddings,
              sequence_length=word_lengths,
              dtype=(tf.float32))
            char_output = tf.concat([output_fw, output_bw], axis=(-1))
            char_output = tf.reshape(char_output, shape=[-1, s[1], 2 * self.char_hidden_size])
        with tf.variable_scope('abb'):
            nen_embedding = tf.layers.dense((self.nen_labels), 5, kernel_initializer=(tf.contrib.layers.xavier_initializer()),
              activation=(tf.nn.tanh))
        with tf.variable_scope('final_embedding'):
            word_embeddings = tf.concat([word_embeddings, pos_embeddings, char_output, nen_embedding], axis=(-1))
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_op)

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope('bi-lstm'):
            cell_fw = tf.contrib.rnn.LSTMCell((self.n_hidden_size), initializer=(tf.contrib.layers.xavier_initializer()))
            cell_bw = tf.contrib.rnn.LSTMCell((self.n_hidden_size), initializer=(tf.contrib.layers.xavier_initializer()))
            output, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, (self.word_embeddings),
              sequence_length=(self.label_lens),
              dtype=(tf.float32))
            lstm_output1 = tf.concat(output, 2)
        with tf.variable_scope('batch_normalization_lstm'):
            lstm_output1 = tf.layers.batch_normalization(lstm_output1, training=(self.is_training))
            lstm_output1 = tf.nn.dropout(lstm_output1, self.dropout_lstm)
        with tf.variable_scope('feedforward_after_lstm'):
            W = tf.get_variable('W', shape=[2 * self.n_hidden_size, self.n_hidden_size], initializer=(tf.contrib.layers.xavier_initializer()))
            b = tf.get_variable('b', shape=[self.n_hidden_size], dtype=(tf.float32), initializer=(tf.zeros_initializer()))
            ntime_steps = tf.shape(lstm_output1)[1]
            lstm_output1 = tf.reshape(lstm_output1, [-1, 2 * self.n_hidden_size])
            outputs = tf.nn.xw_plus_b(lstm_output1, W, b, name='output_before_tanh')
        with tf.variable_scope('batch_normalization_fc'):
            outputs = tf.layers.batch_normalization(outputs, training=(self.is_training))
            outputs = tf.nn.tanh(outputs, name='output_after_tanh')
            outputs = tf.nn.dropout(outputs, self.dropout_op)
        with tf.variable_scope('proj'):
            W = tf.get_variable('W', shape=[self.n_hidden_size, self.num_of_class], dtype=(tf.float32),
              initializer=(tf.contrib.layers.xavier_initializer()))
            b = tf.get_variable('b', shape=[self.num_of_class], dtype=(tf.float32), initializer=(tf.zeros_initializer()))
            pred = tf.nn.xw_plus_b(outputs, W, b, name='output_before_crf')
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.num_of_class])


class BiLSTMCRF(BiLSTMCRFCore):

    def __init__(self, model_name, embeddings, batch_size):
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.session = None

    def load_model(self):
        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_logits_op()
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session, tf.train.latest_checkpoint(self.model_name))

    def _next_batch_predict(self, data, num_batch):
        start = 0
        idx = 0
        while 1:
            if idx < num_batch:
                X_batch = data['X'][start:start + self.batch_size]
                Y_nen_batch = data['Y_nen'][start:start + self.batch_size]
                Z_batch = data['Z'][start:start + self.batch_size]
                char_ids, word_ids = zip(*[zip(*x) for x in X_batch])
                word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0)
                char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
                nen_labels, _ = pad_sequences(Y_nen_batch, pad_tok=([0] * self.nen_label_size), nlevels=3)
                pos_ids, _ = pad_sequences(Z_batch, pad_tok=0)
                start += self.batch_size
                idx += 1
                yield (word_ids, char_ids, nen_labels, sequence_lengths, word_lengths, pos_ids)

    def predict_classes(self, data, transition_params):
        num_batch = len(data['X']) // self.batch_size + 1
        y_pred = []
        for idx, batch in enumerate(self._next_batch_predict(data={'X':data['X'],  'Z':data['Z'],  'Y_nen':data['Y_nen']},
          num_batch=num_batch)):
            words, chars, nen_labels, sequence_lengths, word_lengths, poses = batch
            feed_dict = {self.word_ids: words, 
             self.char_ids: chars, 
             self.word_lengths: word_lengths, 
             self.nen_labels: nen_labels, 
             self.label_lens: sequence_lengths, 
             self.dropout_op: 1.0, 
             self.dropout_lstm: 1.0, 
             self.word_pos_ids: poses, 
             self.is_training: False}
            logits = self.session.run((self.logits), feed_dict=feed_dict)
            for logit, leng in zip(logits, sequence_lengths):
                logit = logit[:leng]
                decode_sequence, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
                y_pred.append(decode_sequence)

        return y_pred