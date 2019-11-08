# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\pre_process\segmenters\spacy.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 1048 bytes
import re
from module.spacy import Spacy
from pre_process.models import Segmenter

class SpacySegmenter(Segmenter):

    def __init__(self):
        super().__init__()

    @staticmethod
    def __fix_segmentation(raw_sentences):
        new_sentences = []
        cur_sent = raw_sentences[0]
        i = 1
        while 1:
            if i < len(raw_sentences):
                if cur_sent[(-1)] in '.!?;':
                    new_sentences.append(cur_sent)
                    cur_sent = raw_sentences[i]
                else:
                    cur_sent += ' ' + raw_sentences[i]
                i += 1

        new_sentences.append(cur_sent)
        return new_sentences

    def segment(self, text):
        """
        :param string text: document that needs to be segmented
        :return: list of string
        """
        text = re.sub('\\(ABSTRACT TRUNCATED AT 250 WORDS\\)', '', text)
        parsed = Spacy.parse(text)
        return self._SpacySegmenter__fix_segmentation([str(sent) for sent in parsed.sents])