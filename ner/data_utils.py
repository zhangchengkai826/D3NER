# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\ner\data_utils.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 3597 bytes
import numpy as np
from train.constants import MAX_SENT_LENGTH

class MyIOError(Exception):

    def __init__(self, filename):
        message = '\n        ERROR: Unable to locate file {}.\n\n        FIX: Have you tried running python build_data.py first?\n        This will build vocab file from your train, test and dev sets and\n        trim your word vectors.'.format(filename)
        super(MyIOError, self).__init__(message)


def limit_sent_length(sentence):
    return sentence[:MAX_SENT_LENGTH]


def cut_incomplete_entity(x, y):
    if len(y) == MAX_SENT_LENGTH:
        if y[(-1)][0] == 'B':
            x = x[:-1]
            y = y[:-1]
        elif y[(-1)][0] == 'I':
            i = len(y) - 1
            while 1:
                if i >= 0:
                    if y[i][0] == 'B':
                        break
                    i -= 1

            x = x[:i]
            y = y[:i]
    return (
     x, y)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return (
     sequence_padded, sequence_length)


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: level option
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    return (sequence_padded, sequence_length)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as (data):
            return data['embeddings']
    except IOError:
        raise MyIOError(filename)


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as (f):
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx + 1

    except IOError:
        raise MyIOError(filename)

    return d