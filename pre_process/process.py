# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: D:\Projects\d3ner\pre_process\process.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 2273 bytes
import models, constants
from pre_process import opt

def __parse_sentence(sent, doc_offset, tokenizer):
    """
    Perform tokenization on sent
    :param string sent:
    :param tuple doc_offset: (start, end)
    :param tokenizer:
    :return: Sentence object
    """
    sent_obj = models.Sentence(content=sent, doc_offset=doc_offset)
    sent_obj.tokens = []
    tokens = tokenizer.parse(sent)
    current_pos = 0
    for tok in tokens:
        t = tok.string.strip()
        start_offset = sent.find(t, current_pos)
        end_offset = start_offset + len(t)
        token_obj = models.Token(content=t,
          sent_offset=(
         start_offset, end_offset),
          doc_offset=(
         doc_offset[0] + start_offset,
         doc_offset[0] + end_offset),
          metadata={'POS': tok.tag_})
        current_pos = end_offset
        sent_obj.tokens.append(token_obj)

    return sent_obj


def process(documents, config, sent_type=constants.SENTENCE_TYPE_GENERAL):
    """
    :param dict documents: format(id => content)
    :param int sent_type: type of sentence
    :param dict config: keys are 'segmenter', 'tokenizer', 'options'
    :return: list of Document
    """
    assert config.__class__.__name__ == 'dict', '"config" must be a dict.'
    doc_objects = []
    segmenter = config.get(opt.SEGMENTER_KEY, opt.SpacySegmenter())
    tokenizer = config.get(opt.TOKENIZER_KEY, opt.SpacyTokenizer())
    for i in documents:
        doc_obj = models.Document(id=i, content=(documents[i]))
        doc_obj.sentences = []
        raw_sentences = segmenter.segment(doc_obj.content)
        current_pos = 0
        for s in raw_sentences:
            start_offset = documents[i].find(s, current_pos)
            end_offset = start_offset + len(s)
            sent_obj = __parse_sentence(s, (start_offset, end_offset), tokenizer)
            sent_obj.type = sent_type
            current_pos = end_offset
            doc_obj.sentences.append(sent_obj)

        doc_objects.append(doc_obj)

    optional = config.get(opt.OPTION_KEY, [])
    for o in optional:
        for doc_obj in doc_objects:
            o.process(doc_obj)

    return doc_objects