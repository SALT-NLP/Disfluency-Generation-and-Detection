from __future__ import division

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch import cuda

import table
import opts
from table.Utils import set_seed
'''import table.Models
import table.ModelConstructor
import table.modules


from path import Path'''


parser = argparse.ArgumentParser(description='train.py')

# opts.py
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()

opt.dataset=opt.dataset+opt.tag_type
opt.data = os.path.join(opt.root_dir, opt.dataset)
opt.save_dir = os.path.join(opt.root_dir, opt.dataset)



def load_fields(train, valid, checkpoint):
    fields = torch.load(opt.data + '/vocab.pt')
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

    return fields


def build_model(model_opt, fields, checkpoint):
    print('Building model...')
    model = table.ModelConstructor.make_base_model(
        model_opt, fields, checkpoint)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = table.Optim(
            opt.optim, opt.learning_rate, opt.alpha, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )

    optim.set_parameters(model.parameters())

    return optim


def main():
    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(os.path.join(opt.data, 'train.pt'))
    valid = torch.load(os.path.join(opt.data, 'valid.pt'))
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.

    model_opt = opt

    # Load fields generated from preprocess phase.
    fields = load_fields(train, valid,None)

    print(train[0].__dict__)
    '''# Build model.
    model = build_model(model_opt, fields, checkpoint)

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, train, valid, fields, optim)'''


if __name__ == "__main__":
    main()
