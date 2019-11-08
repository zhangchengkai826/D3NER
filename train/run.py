# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\train\run.py
# Compiled at: 2018-06-19 19:17:07
# Size of source mod 2**32: 1867 bytes
import argparse
from train.d3ner_model import BiLSTMCRF
from train.dataset import BioCDataset
from ner.data_utils import get_trimmed_glove_vectors
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train new model.')
    parser.add_argument('model', help='the name of the model, i.e: d3ner_cdr')
    parser.add_argument('dataset', help='the name of the dataset that the model will be trained on, i.e: cdr')
    parser.add_argument('train_set', help='path to the training dataset, i.e: data/cdr/cdr_train.txt')
    parser.add_argument('-dev', '--dev_set', help='path to the development dataset, i.e: data/cdr/cdr_dev.txt', default='')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-es', '--early_stopping', help='use early stopping', action='store_true')
    group.add_argument('-e', '--epoch', help='number of epochs to train', type=int, default=27)
    parser.add_argument('-v', '--verbose', help='print training process', action='store_true')
    parser.add_argument('-ds', '--display_step', help='number of steps before displaying', type=int, default=20)
    args = parser.parse_args()
    train = BioCDataset(args.dataset, args.train_set)
    dev = None
    if args.dev_set:
        dev = BioCDataset(args.dataset, args.dev_set)
    embeddings = get_trimmed_glove_vectors('data/{}/embedding_data.npz'.format(args.dataset))
    model = BiLSTMCRF(model_name=(args.model), embeddings=embeddings, batch_size=120, early_stopping=(args.early_stopping),
      display_step=(args.display_step))
    model.load_data(train, dev=dev)
    model.build()
    if not args.early_stopping:
        model.run_train((args.epoch), verbose=(args.verbose))
    else:
        model.run_train(50, verbose=(args.verbose), patience=4)