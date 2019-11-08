# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\writers.py
# Compiled at: 2018-06-19 19:17:07
# Size of source mod 2**32: 1207 bytes


class Writer:

    def __init__(self):
        pass

    def write(self, **kwargs):
        pass


class BioCreativeWriter(Writer):

    def __init__(self):
        super().__init__()

    def write(self, file_name, raw_documents, dict_nern):
        """
        :param string file_name: path/to/file
        :param dict raw_documents: pmid => {a => str, t => str}
        :param dict of list of models.BioEntity dict_nern:
        """
        with open(file_name, 'w', encoding='utf8') as (f):
            for pmid in raw_documents:
                doc = raw_documents[pmid]
                nern = dict_nern[pmid]
                if 't' in doc:
                    f.write('{}|t|{}\n'.format(pmid, doc['t']))
                if 'a' in doc:
                    f.write('{}|a|{}\n'.format(pmid, doc['a']))
                for entity in nern:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(pmid, entity.tokens[0].doc_offset[0], entity.tokens[(-1)].doc_offset[1], entity.content, entity.type))

                f.write('\n')