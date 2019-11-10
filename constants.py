# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\constants.py
# Compiled at: 2018-07-26 11:46:59
# Size of source mod 2**32: 514 bytes
SENTENCE_TYPE_GENERAL = 0
SENTENCE_TYPE_TITLE = 1
SENTENCE_TYPE_ABSTRACT = 2
with open('entity_types') as (f):
    ENTITY_TYPES = f.read().strip().split('\n')
ETYPE_MAP = {}
REV_ETYPE_MAP = {}
for i in range(len(ENTITY_TYPES)):
    ETYPE_MAP[ENTITY_TYPES[i]] = str(i)
    REV_ETYPE_MAP[str(i)] = ENTITY_TYPES[i]

ALL_LABELS = []
for i in ENTITY_TYPES:
    ALL_LABELS.extend(['B' + ETYPE_MAP[i], 'I' + ETYPE_MAP[i], 'L' + ETYPE_MAP[i], 'U' + ETYPE_MAP[i]])

ALL_LABELS.append('O')
MESH_KEY = 'mesh_id'