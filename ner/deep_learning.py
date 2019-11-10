# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\ner\deep_learning.py
# Compiled at: 2018-07-26 00:36:54
# Size of source mod 2**32: 4714 bytes
import numpy as np, pickle
from models import BioEntity
import constants
from ner.models import NER
from ner.d3ner_model import BiLSTMCRF
from ner.data_utils import get_trimmed_glove_vectors, load_vocab

class TensorNer(NER):

    def __init__(self, model_name, data_name):
        super().__init__()
        PATH_MODEL = 'pre_trained_models/{}/'.format(model_name)
        PATH_DATA = 'data/{}/'.format(data_name)
        embeddings = get_trimmed_glove_vectors(PATH_DATA + 'embedding_data.npz')
        self.vocab_words = load_vocab(PATH_DATA + 'all_words.txt')
        self.vocab_chars = load_vocab(PATH_DATA + 'all_chars.txt')
        self.vocab_poses = load_vocab('data/all_pos.txt')
        self.vocab_ab3p = pickle.load(open(PATH_DATA + 'ab3p_tfidf.pickle', 'rb'))
        self.model = BiLSTMCRF(PATH_MODEL, embeddings, batch_size=32)
        self.model.load_model()
        print('Load NER model finished')
        self.transition_params = np.load(PATH_MODEL + 'crf_transition_params.npy')[0]
        self.all_labels = self.model.all_labels
        self.unk = self.model.unk

    def process(self, document):
        X, Z, Y_nen = self._TensorNer__parse_document_to_data(document)
        y_pred = self.model.predict_classes({'X':X,  'Z':Z,  'Y_nen':Y_nen}, self.transition_params)
        entities = self._TensorNer__decode_y_pred(y_pred, document)
        return entities

    def __parse_document_to_data(self, document):
        X, Z, Y_nen = [], [], []
        abb = []
        tfs = None
        if document.id in self.vocab_ab3p:
            unzipped = list(zip(*self.vocab_ab3p[document.id]))
            abb = unzipped[0]
            tfs = unzipped[1:]
        for s in document.sentences:
            x, z, y_nen = self._TensorNer__parse_sentence(s, abb, tfs)
            X.append(x)
            Z.append(z)
            Y_nen.append(y_nen)

        return (X, Z, Y_nen)

    def __parse_sentence(self, sentence, abb, tfs):
        w = []
        p = []
        n = []
        for i in range(len(sentence.tokens)):
            word = self._process_word(sentence.tokens[i].processed_content)
            pos = self.vocab_poses[sentence.tokens[i].metadata['POS']]
            w += [word]
            p += [pos]
            if word in abb:
                idx = abb.index(word)
                n.append([tfs[k][idx] for k in range(len(constants.ENTITY_TYPES))])
            else:
                n.append([0] * len(constants.ENTITY_TYPES))

        return (w, p, n)

    def _process_word(self, word):
        char_ids = []
        if self.vocab_chars is not None:
            for char in word:
                if char in self.vocab_chars:
                    char_ids += [self.vocab_chars[char]]

        if word in self.vocab_words:
            word_id = self.vocab_words[word]
        else:
            word_id = self.vocab_words[self.unk]
        if self.vocab_chars is not None:
            return (char_ids, word_id)
        else:
            return word_id

    @staticmethod
    def __last_index(cur_idx, array):
        c = cur_idx + 1
        while c < len(array):
            if array[c] == array[cur_idx] + 1:
                c += 1
            elif array[c] == array[cur_idx] + 2:
                return c
            else:
                return c - 1

        return c - 1

    def __decode_y_pred(self, y_pred, document):
        entities = []
        for i in range(len(y_pred)):
            j = 0
            while j < len(y_pred[i]):
                e = None
                if self.all_labels[y_pred[i][j]][0] == 'U':
                    e = BioEntity(etype=(constants.REV_ETYPE_MAP[self.all_labels[y_pred[i][j]][1]]), tokens=(document.sentences[i].tokens[j:j + 1]))
                elif self.all_labels[y_pred[i][j]][0] == 'B':
                    l_idx = self._TensorNer__last_index(j, y_pred[i])
                    if self.all_labels[y_pred[i][l_idx]][0] == 'L' and self.all_labels[y_pred[i][l_idx]][1] == self.all_labels[y_pred[i][j]][1]:
                        e = BioEntity(etype=(constants.REV_ETYPE_MAP[self.all_labels[y_pred[i][j]][1]]), tokens=(document.sentences[i].tokens[j:l_idx + 1]))
                    j = l_idx
                j += 1
                if e:
                    entities.append(e)

        return entities