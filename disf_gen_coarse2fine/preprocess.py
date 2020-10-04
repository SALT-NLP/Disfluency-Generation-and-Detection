# -*- coding: utf-8 -*-

import os
import argparse
import torch
from path import Path

import table
import table.IO
import opts
from table.Utils import set_seed

parser = argparse.ArgumentParser(description='preprocess.py')


# **Preprocess Options**

opts.preprocess_opts(parser)

opt = parser.parse_args()
print('flt',opt.include_flt)
print('lower',opt.lower)
print(vars(opt))
set_seed(opt.seed)

opt.dataset=opt.dataset+opt.tag_type
opt.train_anno = os.path.join(opt.root_dir, opt.dataset, opt.train_data)
opt.valid_anno = os.path.join(opt.root_dir, opt.dataset, 'val.txt')
opt.test_anno = os.path.join(opt.root_dir, opt.dataset, 'test.txt')
opt.save_data = os.path.join(opt.root_dir, opt.dataset)


def main():

    print('Preparing training ...')
    fields = table.IO.TableDataset.get_fields(opt)
    print("Building Training...")
    train = table.IO.TableDataset(
        opt.train_anno, fields, opt)
    print(train[0].__dict__)

    if Path(opt.valid_anno).exists():
        print("Building Valid...")
        valid = table.IO.TableDataset(
            opt.valid_anno, fields, opt)
        print('Valid len:', len(valid))
    else:
        valid = None

    if Path(opt.test_anno).exists():
        print("Building Test...")
        test = table.IO.TableDataset(
            opt.test_anno, fields, opt)
        #print(type(test.fields))
    else:
        test = None

    print("Building Vocab...")
    table.IO.TableDataset.build_vocab(train, valid, test, opt)

    print("Saving train/valid/test/fields")

    torch.save(fields, os.path.join(opt.save_data, 'vocab.pt'))
    '''torch.save(table.IO.TableDataset.save_vocab(fields),
               open(, 'wb'))'''
    #train.fields = []
    train.save(os.path.join(opt.save_data, opt.train_data_pt))
    #torch.save(train, open(os.path.join(opt.save_data, 'train.pt'), 'wb'))

    torch.save(opt, os.path.join(opt.save_data, 'preprocess_opt.pt'))

    if Path(opt.valid_anno).exists():
        valid.save(os.path.join(opt.save_data, 'valid.pt'))
        '''valid.fields = []
        torch.save(valid, open(os.path.join(opt.save_data, 'valid.pt'), 'wb'))'''

    if Path(opt.test_anno).exists():
        test.save(os.path.join(opt.save_data, 'test.pt'))
        '''test.fields = []
        torch.save(test, open(os.path.join(opt.save_data, 'test.pt'), 'wb'))'''


if __name__ == "__main__":
    main()
