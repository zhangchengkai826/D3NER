# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\main.py
# Compiled at: 2018-07-26 14:17:34
# Size of source mod 2**32: 1294 bytes
import argparse, pipelines
from pre_process import opt as pp_opt
from ner import opt as ner_opt

def main(model, dataset, input_file, output_file):
    pre_config = {pp_opt.SEGMENTER_KEY: pp_opt.SpacySegmenter(), 
     pp_opt.TOKENIZER_KEY: pp_opt.SpacyTokenizer(), 
     pp_opt.OPTION_KEY: [pp_opt.NumericNormalizer()]}
    nern_config = {ner_opt.NER_KEY: ner_opt.TensorNer(model, dataset)}
    input_file = input_file
    output_file = output_file
    pipeline = pipelines.NerPipeline(input_file, output_file, pre_config=pre_config, nern_config=nern_config)
    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D3NER Program: Recognize biomedical entities in text documents.')
    parser.add_argument('model', help='the name of the model being used, i.e: d3ner_cdr')
    parser.add_argument('dataset', help='the name of the dataset that the model was trained on, i.e: cdr')
    parser.add_argument('input_file', help='path to the input file, i.e: data/cdr/cdr_test.txt')
    parser.add_argument('output_file', help='path to the output file, i.e: output.txt')
    args = parser.parse_args()
    main(args.model, args.dataset, args.input_file, args.output_file)