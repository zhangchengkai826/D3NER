# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\pre_process\tokenizers\spacy.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 652 bytes
import re
from module.spacy import Spacy
from pre_process.models import Tokenizer

class SpacyTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def tokenize(self, sent):
        sent = re.sub('/', ' / ', sent)
        sent = re.sub(']', '] ', sent)
        sent = re.sub('\\s+', ' ', sent)
        tokens = Spacy.parse(sent)
        return [t.string.strip() for t in tokens]

    @staticmethod
    def parse(sent):
        sent = re.sub('/', ' / ', sent)
        sent = re.sub(']', '] ', sent)
        sent = re.sub('\\s+', ' ', sent)
        tokens = Spacy.parse(sent)
        return tokens