# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\module\spacy.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 1268 bytes
import spacy as sp
from spacy.symbols import ORTH, LEMMA
from utils import Timer

class Spacy:
    nlp = None

    @staticmethod
    def load_spacy():
        t = Timer()
        t.start('Load SpaCy', verbal=True)
        Spacy.nlp = sp.load('en_core_web_md')
        t.stop()
        Spacy.nlp.tokenizer.add_special_case('+/-', [{ORTH: '+/-', LEMMA: '+/-'}])
        Spacy.nlp.tokenizer.add_special_case('mg.', [{ORTH: 'mg.', LEMMA: 'mg.'}])
        Spacy.nlp.tokenizer.add_special_case('mg/kg', [{ORTH: 'mg/kg', LEMMA: 'mg/kg'}])
        Spacy.nlp.tokenizer.add_special_case('Gm.', [{ORTH: 'Gm.', LEMMA: 'Gm.'}])
        Spacy.nlp.tokenizer.add_special_case('i.c.', [{ORTH: 'i.c.', LEMMA: 'i.c.'}])
        Spacy.nlp.tokenizer.add_special_case('i.c.v.', [{ORTH: 'i.c.v.', LEMMA: 'i.c.v.'}])
        Spacy.nlp.tokenizer.add_special_case('i.v.', [{ORTH: 'i.v.', LEMMA: 'i.v.'}])
        Spacy.nlp.tokenizer.add_special_case('i.p.', [{ORTH: 'i.p.', LEMMA: 'i.p.'}])

    @staticmethod
    def get_spacy_model():
        if Spacy.nlp is None:
            Spacy.load_spacy()
        return Spacy.nlp

    @staticmethod
    def parse(text):
        if Spacy.nlp is None:
            Spacy.load_spacy()
        return Spacy.nlp(text)