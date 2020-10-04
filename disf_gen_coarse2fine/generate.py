from __future__ import division
import os
import argparse
import torch
import codecs
import glob
from nltk.tokenize import word_tokenize

import table
import table.IO
import opts
from multiprocessing import Pool,Manager
from functools import partial


manager = Manager()

punc_list=['.','?','!',';',':','-','--','(',')','[',']','{','}',"'",'"','`','``',"''",'...',',']

parser = argparse.ArgumentParser(description='generate.py')
opts.translate_opts(parser)
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.dataset = opt.dataset + opt.tag_type
opt.anno = os.path.join(opt.root_dir, opt.dataset, '{}.txt'.format(opt.split))

if opt.beam_size > 0:
    opt.batch_size = 1

print('trans_opt:',vars(opt))

def read_anno(anno_path, opt):
    js_list=[]
    with open(anno_path, "r", encoding='utf-8') as reader:
        for line in reader:
            js_list.append({'src':line.strip().split()})
    return js_list

def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = read_anno(opt.anno, opt)
    print('Finished reading, data_len',len(js_list))
    f_list=None
    for fn_model in glob.glob(opt.model_path):
        opt.model = fn_model
        print(fn_model)
        print(opt.anno)

        translator = table.Translator(opt, dummy_opt.__dict__)
        data = table.IO.TableDataset(
            js_list, translator.fields, translator.model_opt, only_generate=True)
        print('Total number of sents in dataset:', len(data))
        test_data = table.IO.OrderedIterator(
            dataset=data, device=device, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)
        print('Total number of batches in dataset:', len(data) // opt.batch_size)

        # inference
        r_list = []
        idx = 0
        with torch.no_grad():
            for batch in test_data:
                idx += 1
                if idx%5000==0:
                    print('Batch:',idx)
                r = translator.translate(batch,js_list)
                r_list += r

        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
            len(r_list), len(js_list))

        f_list=r_list

        # evaluation
        if opt.eval_diversity:
            for pred, gold in zip(r_list, js_list):
                pred.eval_diversity(gold['src'], pred.disf_frags)
            print('Results:')
            print('disf_less_than_one:', sum((x.disflen_lessthanone for x in r_list)))
            onegrams = []
            twograms = []
            for x in r_list:
                onegrams.extend(x.one_grams)
                twograms.extend(x.two_grams)
            c_correct = sum(onegrams)
            acc = c_correct / len(onegrams)
            print('{}: {} / {} = {:.2%}'.format('onegram',
                                                c_correct, len(onegrams), acc))
            c_correct = sum(twograms)
            acc = c_correct / len(twograms)
            print('{}: {} / {} = {:.2%}'.format('twogram',
                                                c_correct, len(twograms), acc))


    assert f_list is not None
    print("Writing to "+os.path.join(opt.root_dir, opt.dataset, opt.output_file+'_generated'))
    disf_generated=0
    with open(os.path.join(opt.root_dir, opt.dataset, opt.output_file+'_generated'), 'w', encoding='utf-8') as writer:
        for x in f_list:
            if 'I' in x.tgt_tags:
                disf_generated+=1
                assert (len(x.tgt) == len(x.tgt_tags))
                writer.write('\t'.join(x.tgt) + '\n')
                writer.write('\t'.join(['P'] * len(x.tgt_tags)) + '\n')
                writer.write('\t'.join(x.tgt_tags) + '\n')
                writer.write('\n')
    print('Disf_sents/Total_sents',disf_generated,len(f_list))

if __name__ == "__main__":
    main()
