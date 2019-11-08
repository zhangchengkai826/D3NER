# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\pipelines.py
# Compiled at: 2018-06-19 19:17:05
# Size of source mod 2**32: 1497 bytes
import data_managers, readers, pre_process, ner, writers, constants

class Pipeline:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.reader = None
        self.writer = None
        self.data_manager = None

    def run(self):
        pass


class NerPipeline(Pipeline):

    def __init__(self, input_file, output_file, pre_config={}, nern_config={}):
        super().__init__(input_file, output_file)
        self.reader = readers.BioCreativeReader(self.input_file)
        self.writer = writers.BioCreativeWriter()
        self.data_manager = data_managers.CDRDataManager()
        self.pre_config = pre_config
        self.nern_config = nern_config

    def run(self):
        raw_documents = self.reader.read()
        title_docs, abstract_docs = self.data_manager.parse_documents(raw_documents)
        title_doc_objs = pre_process.process(title_docs, self.pre_config, constants.SENTENCE_TYPE_TITLE)
        abs_doc_objs = pre_process.process(abstract_docs, self.pre_config, constants.SENTENCE_TYPE_ABSTRACT)
        doc_objects = self.data_manager.merge_documents(title_doc_objs, abs_doc_objs)
        dict_nern = ner.process(doc_objects, self.nern_config)
        self.writer.write(self.output_file, raw_documents, dict_nern)