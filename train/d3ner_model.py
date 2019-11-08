# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\train\d3ner_model.py
# Compiled at: 2018-06-19 19:17:07
# Size of source mod 2**32: 11982 bytes
import numpy as np, os, tensorflow as tf
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from ner.data_utils import pad_sequences
from ner.d3ner_model import BiLSTMCRFCore
from utils import Timer, Log
from train.constants import TRAINED_MODELS
seed = 13
np.random.seed(seed)

class BiLSTMCRF(BiLSTMCRFCore):

    def __init__(self, model_name, embeddings, batch_size, early_stopping=False, display_step=20):
        super().__init__()
        self.model_name = TRAINED_MODELS + model_name + '/'
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.display_step = display_step
        self.X_train = None
        self.Y_train = None
        self.Z_train = None
        self.Y_nen_train = None
        self.X_dev = None
        self.Y_dev = None
        self.Z_dev = None
        self.Y_nen_dev = None

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('crf_layers'):
            self.transition_params = tf.get_variable('transitions', shape=[
             self.num_of_class, self.num_of_class],
              initializer=(tf.contrib.layers.xavier_initializer()))
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.label_lens, self.transition_params)
            self.loss = tf.reduce_mean(-log_likelihood)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train_step'):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0005, momentum=0.9)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def build(self):
        timer = Timer()
        timer.start('Building model...')
        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_logits_op()
        self._add_loss_op()
        self._add_train_op()
        timer.stop()

    def _accuracy(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_op] = 1.0
        feed_dict[self.dropout_lstm] = 1.0
        feed_dict[self.is_training] = False
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
        f1 = []
        for logit, label, leng in zip(logits, feed_dict[self.labels], feed_dict[self.label_lens]):
            logit = logit[:leng]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
            label = label[:leng]
            f1.append(f1_score(viterbi_sequence, label, average='macro'))

        return np.mean(f1)

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            X_batch = data['X'][start:start + self.batch_size]
            if len(X_batch) == 0:
                break
            Y_batch = data['Y'][start:start + self.batch_size]
            Y_nen_batch = data['Y_nen'][start:start + self.batch_size]
            Z_batch = data['Z'][start:start + self.batch_size]
            char_ids, word_ids = zip(*[zip(*x) for x in X_batch])
            word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            labels, _ = pad_sequences(Y_batch, pad_tok=8)
            nen_labels, _ = pad_sequences(Y_nen_batch, pad_tok=([0] * self.nen_label_size), nlevels=3)
            pos_ids, _ = pad_sequences(Z_batch, pad_tok=0)
            start += self.batch_size
            idx += 1
            yield (word_ids, char_ids, labels, nen_labels, sequence_lengths, word_lengths, pos_ids)

    def _train(self, epochs, patience=4, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)
        saver = tf.train.Saver(max_to_keep=1)
        best_f1 = 0.0
        nepoch_noimp = 0
        with tf.Session() as (sess):
            sess.run(tf.global_variables_initializer())
            num_batch_train = len(self.X_train) // self.batch_size + 1
            for e in range(epochs):
                X_train, Y_train, Y_nen_train, Z_train = shuffle(self.X_train, self.Y_train, self.Y_nen_train, self.Z_train)
                for idx, batch in enumerate(self._next_batch(data={'X':X_train,  'Y':Y_train,  'Y_nen':Y_nen_train, 
                 'Z':Z_train},
                  num_batch=num_batch_train)):
                    words, chars, labels, nen_labels, sequence_lengths, word_lengths, poses = batch
                    feed_dict = {self.word_ids: words, 
                     self.char_ids: chars, 
                     self.labels: labels, 
                     self.nen_labels: nen_labels, 
                     self.label_lens: sequence_lengths, 
                     self.word_lengths: word_lengths, 
                     self.dropout_op: 0.5, 
                     self.dropout_lstm: 0.85, 
                     self.word_pos_ids: poses, 
                     self.is_training: True}
                    _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
                    if idx % self.display_step == 0:
                        Log.log('Iter {} - Loss: {} '.format(idx, loss_train))

                Log.log('End epochs {} '.format(e + 1))
                if self.early_stopping:
                    num_batch_val = len(self.X_dev) // self.batch_size + 1
                    total_f1 = []
                    for idx, batch in enumerate(self._next_batch(data={'X':self.X_dev,  'Y':self.Y_dev,  'Y_nen':self.Y_nen_dev, 
                     'Z':self.Z_dev},
                      num_batch=num_batch_val)):
                        words, chars, labels, nen_labels, sequence_lengths, word_lengths, poses = batch
                        f1 = self._accuracy(sess, feed_dict={self.word_ids: words, 
                         self.char_ids: chars, 
                         self.labels: labels, 
                         self.nen_labels: nen_labels, 
                         self.label_lens: sequence_lengths, 
                         self.word_lengths: word_lengths, 
                         self.word_pos_ids: poses})
                        total_f1.append(f1)

                    f1 = sum(total_f1) / len(total_f1)
                    tran = sess.run([self.transition_params])
                    if f1 > best_f1:
                        saver.save(sess, self.model_name + 'bilstm_tfcrf')
                        np.save(self.model_name + 'crf_transition_params', tran)
                        Log.log('Best F1: {} '.format(f1))
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_f1 = f1
                        nepoch_noimp = 0
                    else:
                        nepoch_noimp += 1
                        Log.log('Number of epochs with no improvement: {}'.format(nepoch_noimp))
                        if nepoch_noimp >= patience:
                            break

            if not self.early_stopping:
                tran = sess.run([self.transition_params])
                saver.save(sess, self.model_name + 'bilstm_tfcrf')
                np.save(self.model_name + 'crf_transition_params', tran)

    def load_data(self, train, dev=None):
        timer = Timer()
        timer.start('Loading data')
        self.X_train = deepcopy(train.X)
        self.Y_train = deepcopy(train.Y)
        self.Z_train = deepcopy(train.Z)
        self.Y_nen_train = deepcopy(train.Y_nen)
        if dev:
            self.X_train.extend(dev.X)
            self.Y_train.extend(dev.Y)
            self.Z_train.extend(dev.Z)
            self.Y_nen_train.extend(dev.Y_nen)
        if self.early_stopping:
            self.X_train, self.X_dev, self.Y_train, self.Y_dev, self.Z_train, self.Z_dev, self.Y_nen_train, self.Y_nen_dev = train_test_split((self.X_train), (self.Y_train),
              (self.Z_train),
              (self.Y_nen_train),
              test_size=0.1,
              random_state=13)
            Log.log('Number of validating examples: {}'.format(len(self.X_dev)))
        Log.log('Number of training examples: {}'.format(len(self.X_train)))
        timer.stop()

    def run_train(self, epochs, verbose=True, patience=5):
        timer = Timer()
        timer.start('Training model...')
        self._train(epochs, verbose=verbose, patience=patience)
        timer.stop()

    def evaluate(self, test, transition_params, argmax=False):
        saver = tf.train.Saver()
        with tf.Session() as (sess):
            print('Testing model over test set')
            saver.restore(sess, tf.train.latest_checkpoint(self.model_name))
            y_true = []
            y_pred = []
            num_batch = len(test.X) // self.batch_size + 1
            for idx, batch in enumerate(self._next_batch(data={'X':test.X,  'Y':test.Y,  'Z':test.Z,  'Y_nen':test.Y_nen}, num_batch=num_batch)):
                words, chars, labels, nen_labels, sequence_lengths, word_lengths, poses = batch
                feed_dict = {self.word_ids: words, 
                 self.char_ids: chars, 
                 self.word_lengths: word_lengths, 
                 self.nen_labels: nen_labels, 
                 self.label_lens: sequence_lengths, 
                 self.dropout_op: 1.0, 
                 self.dropout_lstm: 1.0, 
                 self.word_pos_ids: poses, 
                 self.is_training: False}
                logits = sess.run((self.logits), feed_dict=feed_dict)
                for logit, label, leng in zip(logits, labels, sequence_lengths):
                    logit = logit[:leng]
                    label = label[:leng]
                    if argmax:
                        decode_sequence = np.argmax(logit, axis=(-1))
                    else:
                        decode_sequence, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
                    y_pred.extend(decode_sequence)
                    y_true.extend(label)

            y_pred = np.array(y_pred, dtype=(np.int32))
            y_true = np.array(y_true, dtype=(np.int32))
        return (
         y_pred, y_true)