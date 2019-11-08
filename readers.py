# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\readers.py
# Compiled at: 2018-06-19 19:17:07
# Size of source mod 2**32: 1396 bytes
from collections import defaultdict
import re

class Reader:

    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, **kwargs):
        pass


class BioCreativeReader(Reader):

    def __init__(self, file_name):
        super().__init__(file_name)
        with open(file_name, 'r', encoding='utf8') as (f):
            self.lines = f.readlines()

    def read(self):
        """
        :return: dict of abstract's: {<id>: {'t': <string>, 'a': <string>}}
        """
        regex = re.compile('^([\\d]+)\\|([at])\\|(.+)$', re.U | re.I)
        abstracts = defaultdict(dict)
        for line in self.lines:
            matched = regex.match(line)
            if matched:
                data = matched.groups()
                abstracts[data[0]][data[1]] = data[2]

        return abstracts

    def read_entity(self):
        """
        :return: dict of entity's: {<id>: [(pmid, start, end, content, type, id)]}
        """
        regex = re.compile('^(\\d+)\\t(\\d+)\\t(\\d+)\\t([^\\t]+)\\t(\\S+)\\t(\\S+)', re.U | re.I)
        ret = defaultdict(list)
        for line in self.lines:
            matched = regex.search(line)
            if matched:
                data = matched.groups()
                ret[data[0]].append(tuple([data[0], int(data[1]), int(data[2]), data[3], data[4], data[5]]))

        return ret