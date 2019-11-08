# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\train\dataset.py
# Compiled at: 2018-07-26 00:36:54
# Size of source mod 2**32: 2959 bytes
import pickle, numpy as np
from ner.data_utils import load_vocab, limit_sent_length, cut_incomplete_entity
from train.build_data import parse_raw
from train.constants import UNK
from constants import ALL_LABELS, ENTITY_TYPES
seed = 13
np.random.seed(seed)

class BioCDataset:

    def __init__(self, dataset, data_file):
        self.X = []
        self.Y = []
        self.Z = []
        self.Y_nen = []
        self.data_file = data_file
        self.vocab_words = load_vocab('data/{}/all_words.txt'.format(dataset))
        self.vocab_chars = load_vocab('data/{}/all_chars.txt'.format(dataset))
        self.vocab_poses = load_vocab('data/all_pos.txt')
        self.vocab_ab3p = pickle.load(open('data/{}/ab3p_tfidf.pickle'.format(dataset), 'rb'))
        self._process_data()

    def _process_data(self):
        with open((self.data_file), 'r', encoding='utf8') as (f):
            raw_data = f.readlines()
        data_x, data_y, data_lens, data_pos, pmid = parse_raw(raw_data)
        for i in range(len(data_x)):
            words, labels, nen_labels, poses = ([], [], [], [])
            sent = limit_sent_length(data_x[i])
            sent_labels = limit_sent_length(data_y[i])
            sent, sent_labels = cut_incomplete_entity(sent, sent_labels)
            sent_pos = data_pos[i][:len(sent)]
            if len(data_y[i]) == 0:
                continue
            abb = []
            if pmid[i][0] in self.vocab_ab3p:
                unzipped = list(zip(*self.vocab_ab3p[pmid[i][0]]))
                abb = unzipped[0]
                tfs = unzipped[1:]
            for word, label, pos in zip(sent, sent_labels, sent_pos):
                if word in abb:
                    idx = abb.index(word)
                    nen_label = [tfs[k][idx] for k in range(len(tfs))]
                else:
                    nen_label = [
                     0] * len(ENTITY_TYPES)
                word = self._process_word(word)
                label = ALL_LABELS.index(label)
                pos_ids = self.vocab_poses[pos]
                poses += [pos_ids]
                words += [word]
                labels += [label]
                nen_labels += [nen_label]

            self.X.append(words)
            self.Y.append(labels)
            self.Y_nen.append(nen_labels)
            self.Z.append(poses)

    def _process_word(self, word):
        char_ids = []
        for char in word:
            if char in self.vocab_chars:
                char_ids += [self.vocab_chars[char]]

        if word in self.vocab_words:
            word_id = self.vocab_words[word]
        else:
            word_id = self.vocab_words[UNK]
        return (
         char_ids, word_id)