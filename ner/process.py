# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\ner\process.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 549 bytes
from ner import opt

def process(documents, config):
    """
    Perform NER and NEN
    :param list documents: list of Document objects
    :param dict config: keys are ('ner' and 'nen') or 'ner_nen'
    :return: dict: {"id": <Entity objects>}
    """
    assert config.__class__.__name__ == 'dict', '"config" must be a dict.'
    res = {}
    ner_method = config.get(opt.NER_KEY)
    if ner_method:
        for d in documents:
            entities = ner_method.process(d)
            res[d.id] = entities

    return res