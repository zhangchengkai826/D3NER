# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\train\build_data.py
# Compiled at: 2018-07-26 12:01:55
# Size of source mod 2**32: 16748 bytes
import numpy as np, pickle, re, os, subprocess, argparse
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ner.data_utils import load_vocab
from module.spacy import Spacy
from pre_process.segmenters.spacy import SpacySegmenter
from pre_process.tokenizers.spacy import SpacyTokenizer
from utils import Timer
from constants import ETYPE_MAP, ENTITY_TYPES

class PreProcess:

    def __init__(self):
        self.segmenter = SpacySegmenter()
        self.tokenizer = SpacyTokenizer()
        self.stemmer = SnowballStemmer('english')
        self.simple_tokenizer = lambda doc: doc.split()

    def segment(self, text):
        if text == '':
            return []
        else:
            return self.segmenter.segment(text)

    def tokenize(self, sent):
        sent = self._clean(sent)
        return self.tokenizer.tokenize(sent.strip())

    @staticmethod
    def parse(text):
        text = re.sub('/', ' / ', text)
        text = re.sub(']', '] ', text)
        text = PreProcess._clean(text)
        return Spacy.parse(text.strip())

    @staticmethod
    def normalize(sent):
        for i in range(len(sent)):
            temp = re.sub('[\\d]+', '0', sent[i])
            temp = re.sub('0+', '0', temp)
            temp = re.sub('0.0', '0', temp)
            sent[i] = temp

        return sent

    @staticmethod
    def _clean(sent):
        sent = re.sub('-', ' ', sent)
        sent = re.sub('\\s+', ' ', sent)
        return sent

    @staticmethod
    def separate_entities(text, entities):
        if not entities:
            return text
        else:
            cur_off = 0
            abstract = ''
            for e in entities:
                abstract += text[cur_off:int(e[3])]
                if '.' in e[0]:
                    e[0] = re.sub('\\.', ' ', e[0])
                abstract += ' ' + e[0] + ' '
                cur_off = int(e[4])

            abstract += text[cur_off:]
            return re.sub('\\s+', ' ', abstract.strip())

    def abb_nen_special(self, term, stem=False):
        ret = re.sub('[-, ]+', ' ', term).lower()
        if stem:
            w = self.simple_tokenizer(ret)
            w = [self.stemmer.stem(tk) for tk in w]
            ret = ' '.join(w)
        return ret


class TFIDFNEN:

    def __init__(self, train_dicts, stem=False):
        self.stem = stem
        self._train_id_dict = self.get_all_concepts(train_dicts)
        self.train_documents = []
        for i in self._train_id_dict.values():
            self.train_documents.extend(i)

        self.tf_idf = None
        self.trained_concepts = None

    def _make_dicts(self, raw_data, id_dict_to_set=True):
        pp = PreProcess()
        id_dict = {}
        for d in raw_data:
            c = pp.abb_nen_special((d[1]), stem=(self.stem))
            if id_dict_to_set:
                if not id_dict.get(d[0]):
                    id_dict[d[0]] = set()
                id_dict[d[0]].add(c)
            elif not id_dict.get(d[0]):
                id_dict[d[0]] = []
            else:
                id_dict[d[0]].append(c)

        return id_dict

    def get_all_concepts(self, dicts):
        raw_data = []
        for d in dicts:
            for k, v in d.items():
                for n in v:
                    raw_data.append([k, n])

        id_dict = self._make_dicts(raw_data, id_dict_to_set=True)
        return id_dict

    def train(self):
        self.tf_idf = TfidfVectorizer(norm='l2', use_idf=True,
          smooth_idf=False,
          tokenizer=None,
          analyzer='char_wb',
          ngram_range=(1, 3))
        self.trained_concepts = self.tf_idf.fit_transform(list(self.train_documents))


def parse_raw(raw_data):
    abstract_text = ''
    entities = []
    all_label = []
    all_pos = []
    all_sentence = []
    all_input_lens = []
    all_pmid = []
    pmid = None
    pre_process = PreProcess()
    for line in raw_data:
        splitted = line.strip().split('\t')
        if len(splitted) < 2:
            if '|' in line:
                abstract_text += line.strip().split('|')[2] + ' '
                pmid = int(line.strip().split('|')[0])
        elif len(splitted) >= 5:
            e = splitted[3:6] + splitted[1:3]
            e[1] = ETYPE_MAP[e[1]]
            entities.append(e)
        elif 'CID' not in splitted:
            abstract_text = pre_process.separate_entities(abstract_text, entities)
            sentences = pre_process.segment(abstract_text.strip())
            cur_entity_idx = 0
            for sent in sentences:
                words_sp = pre_process.parse(sent)
                words = [w.string.strip() for w in words_sp]
                labels = [
                 'O'] * len(words)
                pos = [w.tag_ for w in words_sp]
                indexes = [
                 pmid] * len(words)
                i = 0
                while i < len(words):
                    if cur_entity_idx < len(entities):
                        tokens = pre_process.tokenize(entities[cur_entity_idx][0])
                        window = words[i:i + len(tokens)]
                        if window == tokens:
                            if len(tokens) == 1:
                                labels[i] = 'U' + entities[cur_entity_idx][1]
                            else:
                                labels[i] = 'B' + entities[cur_entity_idx][1]
                                labels[i + len(tokens) - 1] = 'L' + entities[cur_entity_idx][1]
                                labels[i + 1:i + len(tokens) - 1] = ['I' + entities[cur_entity_idx][1]] * (len(tokens) - 2)
                            i += len(tokens)
                            cur_entity_idx += 1
                        else:
                            i += 1

                pre_process.normalize(words)
                all_label.append(labels)
                all_sentence.append(words)
                all_input_lens.append(len(labels))
                all_pos.append(pos)
                all_pmid.append(indexes)

            entities = []
            abstract_text = ''

    return (
     all_sentence, all_label, all_input_lens, all_pos, all_pmid)


def get_vocabs(full_data):
    vocab_word = set()
    vocab_char = set()
    raw_data = full_data.split('\n')
    data_x, data_y, data_lens, _, _ = parse_raw(raw_data)
    for sen in data_x:
        vocab_word.update(sen)
        for word in sen:
            vocab_char.update(word)

    print('- done. {} tokens'.format(len(vocab_word)))
    return (
     vocab_word, vocab_char)


def get_embedding_vocab(we_file):
    vocab = set()
    with open(we_file, 'rb') as (f):
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while 1:
                ch = f.read(1)
                if ch == b' ':
                    word = (b'').join(word)
                    break
                if ch != b'\n':
                    word.append(ch)

            word = word.decode('utf-8')
            f.read(binary_len)
            vocab.add(word)

    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print('Writing vocab...')
    vocab = list(vocab)
    vocab.sort()
    with open(filename, 'w') as (f):
        for i, word in enumerate(vocab):
            f.write('{}\n'.format(word))

        f.write('$UNK$')
    print('- done. {} tokens'.format(len(vocab)))


def export_trimmed_glove_vectors(we_file, vocab, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        we_file: path to word embedding model
        vocab: dictionary vocab[word] = index
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab) + 1, dim])
    with open(we_file, 'rb') as (f):
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while 1:
                ch = f.read(1)
                if ch == b' ':
                    word = (b'').join(word)
                    break
                if ch != b'\n':
                    word.append(ch)

            word = word.decode('utf-8')
            if word in vocab:
                embedding = np.fromstring((f.read(binary_len)), dtype='float32')
                word_idx = vocab[word]
                embeddings[word_idx] = embedding
            else:
                f.read(binary_len)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


FULL_ABS = os.path.join(os.getcwd(), 'tmp', 'full_abstract')
FULL_NEW = os.path.join(os.getcwd(), 'tmp', 'full_new')
AB3P_RAW = os.path.join(os.getcwd(), 'tmp', 'ab3p')
KNOWLEDGE_BASE = {'Disease':os.path.join('data', 'ctd', 'ctd_disease_id_term.txt'), 
 'Chemical':os.path.join('data', 'ctd', 'ctd_chemical_id_term.txt')}

def make_ab3p(dataset, full_data, ab3p_path):
    full_data = full_data.split('\n')
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    abstract_text = ''
    with open('{}_{}'.format(FULL_ABS, dataset), 'w') as (f):
        i = 0
        for line in full_data:
            splitted = line.strip().split('\t')
            if len(splitted) < 2:
                if '|' in line:
                    doc_id = int(line.strip().split('|')[0])
                    abstract_text += line.strip().split('|')[2] + ' '
            else:
                if len(splitted) >= 5:
                    continue
                if 'CID' not in splitted:
                    i += 1
                    f.write(str(doc_id) + '|' + abstract_text)
                    f.write('\n\n')
                    abstract_text = ''

    fn = open('{}_{}'.format(FULL_NEW, dataset), 'w')
    subprocess.call([os.path.join(ab3p_path, 'identify_abbr'), '{}_{}'.format(FULL_ABS, dataset)], cwd=ab3p_path,
      stdout=fn,
      stderr=(subprocess.DEVNULL))
    with open('{}_{}'.format(FULL_NEW, dataset), 'r') as (fn):
        track_w = defaultdict()
        doc_id = 0
        for line in fn:
            doc = line.strip().split('|')
            if len(doc) == 2:
                doc_id = int(doc[0])
                replace_word = []
            elif len(doc) == 3:
                replace_word.append((doc[0], doc[1]))
            elif len(replace_word) > 0:
                track_w[doc_id] = replace_word
            doc_id = 0
            replace_word = []

    pickle.dump(track_w, open('{}_{}'.format(AB3P_RAW, dataset), 'wb'), pickle.HIGHEST_PROTOCOL)


def read_concepts_from_file(file_name):
    with open(file_name, 'r', encoding='utf8') as (f):
        raw_data = [l.split('\t') for l in f.read().strip().split('\n') if l]
    id_dict = {}
    for d in raw_data:
        if not id_dict.get(d[0]):
            id_dict[d[0]] = []
        id_dict[d[0]].append(d[1])

    return id_dict


def make_cdr_id_term(raw_data):

    def add_to_dict(dik, key, val):
        if not dik.get(key):
            dik[key] = set()
        dik[key].add(val)

    lines = raw_data.split('\n')
    dicts = {}
    for e in ENTITY_TYPES:
        dicts[ETYPE_MAP[e]] = {}

    regex = re.compile('^(\\d+)\\t(\\d+)\\t(\\d+)\\t([^\\t]+)\\t(\\S+)\\t(\\S+)', re.U | re.I)
    for l in lines:
        matched = regex.match(l)
        if matched:
            data = matched.groups()
            c = data[3]
            tp = data[4]
            i = data[5]
            if i == '-1':
                pass
            else:
                i = i.split('|')
                for j in i:
                    if j == '-1':
                        continue
                    add_to_dict(dicts[ETYPE_MAP[tp]], j, c)

    return dicts


def make_ab3p_tfidf(dataset, raw_train_dev, knowledge_base):
    ab3p_doc = pickle.load(open('{}_{}'.format(AB3P_RAW, dataset), 'rb'))
    train_dev_dicts = make_cdr_id_term(raw_train_dev)
    key_list = list(train_dev_dicts.keys())
    key_list.sort()
    tf_idf_nen = {}
    for k in key_list:
        e_type = ENTITY_TYPES[int(k)]
        if knowledge_base.get(e_type):
            print('Knowledge base is available for {}'.format(e_type))
            kb = read_concepts_from_file(knowledge_base[e_type])
            tf_idf_nen[k] = TFIDFNEN([train_dev_dicts[k], kb])
        else:
            tf_idf_nen[k] = TFIDFNEN([train_dev_dicts[k]])
        tf_idf_nen[k].train()

    ab3p_tf = defaultdict()
    j = 0
    for t in ab3p_doc:
        j += 1
        nen_in_doc = []
        ab3p, full = zip(*ab3p_doc[t])
        ab3p = list(ab3p)
        tfs = {}
        cosins = {}
        for k in key_list:
            tfs[k] = tf_idf_nen[k].tf_idf.transform(list(full))
            cosins[k] = linear_kernel(tfs[k], tf_idf_nen[k].trained_concepts)

        for i in range(len(full)):
            tmp = [
             ab3p[i]]
            for k in key_list:
                tmp.append(np.max(cosins[k][i]))

            nen_in_doc.append(tuple(tmp))

        ab3p_tf[t] = nen_in_doc

    pickle.dump(ab3p_tf, open('data/{}/ab3p_tfidf.pickle'.format(dataset), 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build necessary data for model training and evaluating.')
    parser.add_argument('dataset', help='the name of the dataset that the model will be trained on, i.e: cdr')
    parser.add_argument('train_set', help='path to the training dataset, i.e: data/cdr/cdr_train.txt')
    parser.add_argument('word_embedding', help='path to the word embedding pre-trained model, i.e: pre_trained_models/wikipedia-pubmed-and-PMC-w2v.bin')
    parser.add_argument('ab3p', help='path to the Ab3P program.')
    parser.add_argument('-dev', '--dev_set', help='path to the development dataset, i.e: data/cdr/cdr_dev.txt', default='')
    parser.add_argument('-test', '--test_set', help='path to the test dataset, i.e: data/cdr/cdr_test.txt', default='')
    args = parser.parse_args()
    time = Timer()
    time.start('Prepare the data')
    with open(args.train_set, 'r') as (train):
        train_raw = train.read()
    dev_raw = ''
    test_raw = ''
    if args.dev_set:
        with open(args.dev_set, 'r') as (dev):
            dev_raw = dev.read()
    if args.test_set:
        with open(args.test_set, 'r') as (test):
            test_raw = test.read()
    full_raw = train_raw + dev_raw + test_raw
    train_dev = train_raw + dev_raw
    data_vocab_words, data_vocab_char = get_vocabs(full_raw)
    embed_vocab_words = get_embedding_vocab(args.word_embedding)
    vocab_words = data_vocab_words & embed_vocab_words
    write_vocab(vocab_words, 'data/' + args.dataset + '/all_words.txt')
    write_vocab(data_vocab_char, 'data/' + args.dataset + '/all_chars.txt')
    vocab_words = load_vocab('data/' + args.dataset + '/all_words.txt')
    export_trimmed_glove_vectors(we_file=(args.word_embedding), vocab=vocab_words,
      trimmed_filename=('data/' + args.dataset + '/embedding_data'),
      dim=200)
    make_ab3p(args.dataset, full_raw, args.ab3p)
    make_ab3p_tfidf(args.dataset, train_dev, KNOWLEDGE_BASE)
    time.stop()
    print("You can delete the generated 'tmp/' folder.")