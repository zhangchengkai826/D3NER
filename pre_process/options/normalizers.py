# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\pre_process\options\normalizers.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 629 bytes
import re
from pre_process.models import Option

class NumericNormalizer(Option):

    def __init__(self, replace='0'):
        super().__init__()
        self.replace = replace

    def process(self, doc_obj):
        for s in doc_obj.sentences:
            for t in s.tokens:
                temp = re.sub('[\\d]+', self.replace, t.processed_content)
                temp = re.sub('0+', self.replace, temp)
                temp = re.sub('0.0', self.replace, temp)
                temp = re.sub('-', ' ', temp)
                temp = re.sub('\\s+', ' ', temp)
                t.processed_content = temp