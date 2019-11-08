# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\data_managers.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 1654 bytes


class DataManager:

    def __init__(self):
        pass


class CDRDataManager(DataManager):

    @staticmethod
    def parse_documents(raw_documents):
        titles = {}
        abstracts = {}
        for i in raw_documents:
            if raw_documents[i].get('t'):
                titles[i] = raw_documents[i]['t']
            if raw_documents[i].get('a'):
                abstracts[i] = raw_documents[i]['a']

        return (titles, abstracts)

    @staticmethod
    def merge_documents(titles, abstracts):
        """
        :param list titles: list of Document objects
        :param list abstracts: list of Document objects
        :return: list of Document objects
        """
        documents = []
        for t_doc in titles:
            for a_doc in abstracts:
                if t_doc.id == a_doc.id:
                    add_amount = len(t_doc.content) + 1
                    for s in a_doc.sentences:
                        s.doc_offset_add(add_amount)
                        for t in s.tokens:
                            t.doc_offset_add(add_amount)

                    t_doc.content += ' ' + a_doc.content
                    t_doc.sentences += a_doc.sentences
                    abstracts.remove(a_doc)
                    break

            documents.append(t_doc)

        documents += abstracts
        return documents