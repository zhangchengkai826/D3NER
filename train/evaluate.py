# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\train\evaluate.py
# Compiled at: 2018-07-26 00:36:54
# Size of source mod 2**32: 6354 bytes
import numpy as np, argparse
from sklearn.metrics import confusion_matrix, classification_report
from train.d3ner_model import BiLSTMCRF
from train.dataset import BioCDataset
from ner.data_utils import get_trimmed_glove_vectors
from constants import ALL_LABELS, ENTITY_TYPES, ETYPE_MAP

def report_confusion_matrix(true_y, pred_y, all_labels, write_to_file=False, file_name=None):

    def confusion_matrix_to_string(cf_matrix, names):
        text = '\nConfusion Matrix\n'
        text += '\t' + ''.join([n.ljust(8) for n in names]) + '\n'
        for k in range(len(cf_matrix)):
            line = '{}:\t'.format(names[k])
            line += ''.join([str(x).ljust(8) for x in cf_matrix[k]])
            text += line + '\n'

        return text

    target_names = []
    labels = []
    for j in range(len(all_labels)):
        if j in pred_y:
            target_names.append(all_labels[j])
            labels.append(j)

    cf = confusion_matrix(true_y, pred_y, labels=labels)
    if not write_to_file:
        print('\n')
        print(classification_report(true_y, pred_y, labels=labels, target_names=target_names))
        print(confusion_matrix_to_string(cf, target_names))
    else:
        with open(file_name, 'w') as (f):
            f.write(classification_report(true_y, pred_y, labels=labels, target_names=target_names))
            f.write(confusion_matrix_to_string(cf, target_names))


class Evaluator:

    def __init__(self, true_y, pred_y):
        self.y_true = true_y
        self.y_pred = pred_y
        self.true_pos = Evaluator._make_lookup_dict()
        self.total_true = Evaluator._make_lookup_dict()
        self.total_pred = Evaluator._make_lookup_dict()
        self.p = Evaluator._make_lookup_dict()
        self.r = Evaluator._make_lookup_dict()
        self.f1 = Evaluator._make_lookup_dict()
        self.key_list = list(self.true_pos.keys())
        self.key_list.sort()

    @staticmethod
    def _make_lookup_dict():
        v = {}
        for e in ENTITY_TYPES:
            v[ETYPE_MAP[e]] = 0

        return v

    @staticmethod
    def _index_of_L(cur_idx, array):
        """
        :param cur_idx: current index of B in array
        :param array: y_true or y_pred
        """
        j = cur_idx + 1
        while j < len(array):
            if array[j] == array[cur_idx] + 1:
                j += 1
            elif array[j] == array[cur_idx] + 2:
                return j
            else:
                return j - 1

        return j - 1

    def _count_true_pos_strict(self, etype):
        """
        :param etype: "1"|"2"|...
        """
        true_pos = 0
        j = 0
        while j < len(self.y_true):
            if self.y_true[j] == self.y_pred[j]:
                if self.y_pred[j] == ALL_LABELS.index('U' + etype):
                    true_pos += 1
                elif self.y_pred[j] == ALL_LABELS.index('B' + etype):
                    L_idx = self._index_of_L(j, self.y_true)
                    check = self.y_true[j:L_idx + 1] == self.y_pred[j:L_idx + 1]
                    if check.all():
                        true_pos += 1
                    j = L_idx
            j += 1

        return true_pos

    def _count_entities(self, array, etype):
        """
        :param array: y_true or y_pred
        :param etype: "1"|"2"|...
        """
        count = 0
        j = 0
        while j < len(array):
            if array[j] == ALL_LABELS.index('U' + etype):
                count += 1
            elif array[j] == ALL_LABELS.index('B' + etype):
                L_idx = self._index_of_L(j, array)
                if array[L_idx] == ALL_LABELS.index('L' + etype):
                    count += 1
                j = L_idx
            j += 1

        return count

    def evaluate(self):
        for k in self.key_list:
            self.true_pos[k] = self._count_true_pos_strict(k)
            self.total_true[k] = self._count_entities(self.y_true, k)
            self.total_pred[k] = self._count_entities(self.y_pred, k)
            try:
                self.p[k] = self.true_pos[k] / self.total_pred[k] * 100
                self.r[k] = self.true_pos[k] / self.total_true[k] * 100
                self.f1[k] = 2 * self.p[k] * self.r[k] / (self.p[k] + self.r[k])
            except Exception:
                self.p[k] = self.r[k] = self.f1[k] = 0

    def report(self):

        def limit_decimal(num):
            return '{:.2f}'.format(num).ljust(8)

        text = '\n\t' + ''.join([n.ljust(8) for n in ('P', 'R', 'F1')]) + '\n'
        for k in self.key_list:
            text += '{}:\t{}{}{}\n'.format(k, limit_decimal(self.p[k]), limit_decimal(self.r[k]), limit_decimal(self.f1[k]))

        text += '\n'
        text += '\t' + ''.join([n.ljust(8) for n in ('TP', 'True', 'Pred')]) + '\n'
        for k in self.key_list:
            text += '{}:\t{}\t{}\t{}\n'.format(k, self.true_pos[k], self.total_true[k], self.total_pred[k])

        return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model.')
    parser.add_argument('model', help='the name of the model being used, i.e: d3ner_cdr')
    parser.add_argument('dataset', help='the name of the dataset that the model will be trained on, i.e: cdr')
    parser.add_argument('test_set', help='path to the test dataset, i.e: data/cdr/cdr_test.txt')
    parser.add_argument('-cf', '--confusion_matrix', help='report confusion matrix', action='store_true')
    args = parser.parse_args()
    test = BioCDataset(args.dataset, args.test_set)
    embeddings = get_trimmed_glove_vectors('data/{}/embedding_data.npz'.format(args.dataset))
    model = BiLSTMCRF(model_name=(args.model), embeddings=embeddings, batch_size=128)
    model.build()
    transition_params = np.load('pre_trained_models/{}/crf_transition_params.npy'.format(args.model))
    tran = transition_params[0]
    y_pred, y_true = model.evaluate(test, tran)
    evaluator = Evaluator(y_true, y_pred)
    evaluator.evaluate()
    print(evaluator.report())
    if args.confusion_matrix:
        report_confusion_matrix(y_true, y_pred, ALL_LABELS)