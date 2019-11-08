# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\models.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 2174 bytes
import constants

class Token:

    def __init__(self, content=None, doc_offset=None, sent_offset=None, metadata=None):
        self.content = content
        self.doc_offset = doc_offset
        self.sent_offset = sent_offset
        self.processed_content = content
        self.metadata = {} if metadata is None else metadata

    def doc_offset_add(self, amount):
        new_offset = tuple(i + amount for i in self.doc_offset)
        self.doc_offset = new_offset


class Sentence:
    r"""'\n    :type type: int\n    '"""

    def __init__(self, etype=constants.SENTENCE_TYPE_GENERAL, content=None, doc_offset=None, tokens=None, metadata=None):
        self.type = etype
        self.content = content
        self.doc_offset = doc_offset
        self.tokens = tokens
        self.metadata = {} if metadata is None else metadata

    def doc_offset_add(self, amount):
        new_offset = tuple(i + amount for i in self.doc_offset)
        self.doc_offset = new_offset


class Document:

    def __init__(self, id=None, content=None, sentences=None, metadata=None):
        self.id = id
        self.content = content
        self.sentences = sentences
        self.metadata = {} if metadata is None else metadata


class Entity:
    r"""'\n    :type tokens: list of Token.__name__\n    '"""

    def __init__(self, etype='General', tokens=None):
        self.type = etype
        self.tokens = tokens
        self.content = self._make_content()

    def _make_content(self):
        if not self.tokens:
            return
        else:
            content = ''
            cur_offset = self.tokens[0].sent_offset[0]
            for t in self.tokens:
                if t.sent_offset[0] != cur_offset:
                    num_space = t.sent_offset[0] - cur_offset
                    content += ' ' * num_space
                    cur_offset += num_space
                content += t.content
                cur_offset += len(t.content)

            return content


class BioEntity(Entity):

    def __init__(self, etype='General', tokens=None, ids=None):
        super(BioEntity, self).__init__(etype, tokens)
        self.ids = {} if ids is None else ids