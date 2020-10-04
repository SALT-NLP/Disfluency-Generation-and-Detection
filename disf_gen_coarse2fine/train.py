from __future__ import division

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch import cuda

import table
import table.Models
import table.ModelConstructor
import table.modules
from table.Utils import set_seed
import opts
from path import Path

torch.autograd.set_detect_anomaly(True)

def get_save_index(save_dir):
    save_index = 0
    while True:
        if Path(os.path.join(save_dir, 'run.%d' % (save_index,))).exists():
            save_index += 1
        else:
            break
    print('Save in run.',str(save_index))
    return save_index


parser = argparse.ArgumentParser(description='train.py')

# opts.py
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()

opt.dataset=opt.dataset+opt.tag_type
opt.data = os.path.join(opt.root_dir, opt.dataset)
opt.save_dir = os.path.join(opt.root_dir, opt.dataset)
opt.save_path = os.path.join(opt.save_dir, 'run.%d' %
                             (get_save_index(opt.save_dir),))
Path(opt.save_path).mkdir_p()

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")

json.dump(opt.__dict__, open(os.path.join(
    opt.save_path, 'opt.json'), 'w'), sort_keys=True, indent=2)

#cuda.set_device(opt.gpuid[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(opt.seed)

# Set up the logging server.
# logger = Logger(os.path.join(opt.save_path, 'tb'))


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == 0:#-1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = table.Statistics(0, 0, 0, {}, 0)

    return report_stats


def train_model(model, train_data, valid_data, test_data, fields, optim):
    train_iter = table.IO.OrderedIterator(
        dataset=train_data, batch_size=opt.batch_size, device=device, repeat=False)
    valid_iter = table.IO.OrderedIterator(
        dataset=valid_data, batch_size=opt.batch_size, device=device, train=False, sort=True, sort_within_batch=False)
    test_iter = table.IO.OrderedIterator(
        dataset=test_data, batch_size=opt.batch_size, device=device, train=False, sort=True, sort_within_batch=False)
    #print('Test data fields',type(test_iter.dataset.fields))
    #print('Test data fields', type(test_data.fields))

    train_loss = None
    valid_loss = None
    if not opt.no_disf_trans:
        assert (fields['tgt_loss'].vocab.stoi[table.IO.UNK_WORD] == table.IO.UNK)
        assert (fields['src'].vocab.stoi[table.IO.UNK_WORD] == table.IO.UNK)
        assert (fields['tgt'].vocab.stoi[table.IO.UNK_WORD] == table.IO.UNK)
        assert (fields['src_label'].vocab.stoi[table.IO.UNK_WORD] == table.IO.UNK)
        assert (len(fields['src_label'].vocab.stoi) == 4)
        assert (fields['src'].vocab.stoi[table.IO.PAD_WORD] == table.IO.PAD)
        assert (fields['src_label'].vocab.stoi[table.IO.PAD_WORD] == table.IO.PAD)
        assert (fields['tgt_loss'].vocab.stoi[table.IO.PAD_WORD] == table.IO.PAD)
        assert (fields['tgt_loss'].vocab.stoi[table.IO.PAD_WORD] == table.IO.PAD)

        train_loss = table.Loss.LossCompute(fields['tgt_loss'].vocab, model.opt, fields, unk_index=table.IO.UNK,
                                            ignore_index=fields['tgt_loss'].vocab.stoi[table.IO.PAD_WORD],
                                            smooth_eps=model.opt.smooth_eps).to(device)

        valid_loss = table.Loss.LossCompute(fields['tgt_loss'].vocab, model.opt, fields, unk_index=table.IO.UNK,
                                            ignore_index=fields['tgt_loss'].vocab.stoi[table.IO.PAD_WORD],
                                            smooth_eps=model.opt.smooth_eps).to(device)

    train_seg_loss = None
    valid_seg_loss = None
    if opt.disf_seg:
        train_seg_loss = nn.NLLLoss(
            reduction='sum', ignore_index=fields['sent_tag'].vocab.stoi[table.IO.PAD_WORD])
        valid_seg_loss = nn.NLLLoss(
            reduction='sum', ignore_index=fields['sent_tag'].vocab.stoi[table.IO.PAD_WORD])

    #print(model.opt)

    trainer = table.Trainer(model, train_iter, valid_iter,
                            train_loss, valid_loss, train_seg_loss, valid_seg_loss, optim)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        
        if epoch<opt.start_layout_loss or epoch>opt.stop_layout_loss:
            #print('in')
            assert(opt.no_connection_decoder and ((opt.no_share_emb_layout_encoder and opt.seprate_encoder) or (opt.no_connection_encdec and opt.no_attention)))
            for param in model.lay_encoder.parameters():
                param.requires_grad = False
            for param in model.lay_encoder.parameters():
                param.grad=None
            for param in model.lay_classifier.parameters():
                param.requires_grad = False
            for param in model.lay_classifier.parameters():
                param.grad = None
        else:
            for param in model.lay_encoder.parameters():
                param.requires_grad = True
            for param in model.lay_classifier.parameters():
                param.requires_grad = True
        
        #test_stats = trainer.validate(epoch, fields, test_iter)
        #print('Test accuracy: %s' % test_stats.accuracy(True))
        
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, fields, report_func, test_iter)
        print('Train accuracy: %s' % train_stats.accuracy(True))

        # 2. Validate on the validation set.
        valid_stats = trainer.validate(epoch, fields, valid_iter)
        #valid_stats = trainer.validate(epoch, fields, writeFile='temperary_valid')
        print('Validation accuracy: %s' % valid_stats.accuracy(True))

        test_stats = trainer.validate(epoch, fields, test_iter)
        print('Test accuracy: %s' % test_stats.accuracy(True))

        #compare_results('temperary_valid', os.path.join(opt.data, 'val.txt'))

        # 3. Log to remote server.
        # train_stats.log("train", logger, optim.lr, epoch)
        # valid_stats.log("valid", logger, optim.lr, epoch)

        # 4. Update the learning rate
        #if not opt.warm_up:
        trainer.epoch_step(None, epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(opt, epoch, fields, valid_stats)

def compare_results(pred_file, gold_file):
    js_list=[]
    with open(gold_file, "r", encoding='utf-8') as reader:
        i = -1
        for line in reader:
            i += 1
            if i % 4 == 0:
                js_list.append({'sent':[token for token in line.strip().split()]})
            elif i % 4 == 1:
                continue
            if i % 4 == 2:
                l = line.strip().split()
                assert (len(l) == len(js_list[-1]['sent']))
                assert (len(l) > 0)
                js_list[-1]['sent_tag'] = l
    pred_list=[]
    with open(pred_file, "r", encoding='utf-8') as reader:
        i = -1
        for line in reader:
            i += 1
            if i % 4 == 0:
                continue
            elif i % 4 == 1:
                continue
            if i % 4 == 2:
                pred_list.append(line.strip().split())
        '''for line in reader:
            pred_list.append(line.strip().split())'''
    print(len(pred_list),len(js_list))
    assert(len(pred_list)==len(js_list))
    editCorrectTotal = 0
    editTrueTotal = 0
    editPredTotal = 0
    for pred_sent,dic in zip(pred_list,js_list):
        assert(len(pred_sent)==len(dic['sent_tag']))
        #print(pred_sent)
        for pred, gold in zip(pred_sent, dic['sent_tag']):
            if pred=='I' and gold=='I':
                editCorrectTotal += 1
                editTrueTotal += 1
                editPredTotal += 1
            else:
                if pred=='I':
                    editPredTotal += 1
                if gold=='I':
                    editTrueTotal += 1
    p = editCorrectTotal / editPredTotal
    r = editCorrectTotal / editTrueTotal
    f = 2 * p * r / (p + r)
    print("Edit word precision: %f recall: %f fscore: %f" % (p, r, f))

def set_embeddings(fields, pretrain_fields,pretrain_model):
    for k, f in pretrain_fields.items():
        if 'vocab' in f.__dict__:
            print('Setting field',k,'embedding ...')
            assert(k in fields)
            if k=='sent' or k=='src':
                fields[k].vocab.set_vectors(f.vocab.stoi, pretrain_model['q_encoder.embeddings.weight'], pretrain_model['q_encoder.embeddings.weight'].size(1))
                #print('v2', fields[k].vocab.vectors)
            if k=='tgt':
                fields[k].vocab.set_vectors(f.vocab.stoi, pretrain_model['tgt_embeddings.weight'],
                                            pretrain_model['tgt_embeddings.weight'].size(1))
            if k=='sent_tag' or k=='src_label':
                fields[k].vocab=f.vocab
    #print('v1', fields['sent'].vocab.vectors)
    #print('v2', fields['src'].vocab.vectors)
    #print('v3', fields['tgt'].vocab.vectors)


def load_fields(train, valid, test, checkpoint):
    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        if opt.train_from_pretrain:
            fields = torch.load(opt.data + '/vocab.pt')
            set_embeddings(fields,checkpoint['vocab'],checkpoint['model'])
        else:
            fields = checkpoint['vocab']
    else:
        fields = torch.load(opt.data + '/vocab.pt')
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields
    test.fields = fields
    return fields


def build_model(model_opt, fields, checkpoint):
    print('Building model...')
    model = table.ModelConstructor.make_base_model(
        model_opt, fields, checkpoint)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from and not opt.train_from_pretrain:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = table.Optim(
            opt.optim, opt.learning_rate  if not opt.encoder_type=='transformer' else opt.transformer_learning_rate,
            opt.alpha, opt.max_grad_norm, opt.transformer_dim if opt.encoder_type=='transformer' else opt.rnn_size,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            warm_up_step=opt.warm_up_step, warm_up_factor=opt.warm_up_factor,
            opt=opt
        )

    optim.set_parameters(model.parameters())

    return optim


def main():
    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(os.path.join(opt.data, opt.train_data_pt))
    valid = torch.load(os.path.join(opt.data, 'valid.pt'))
    test = torch.load(os.path.join(opt.data, 'test.pt'))

    pre_opt=torch.load(os.path.join(opt.data, 'preprocess_opt.pt'))
    opt.decoder_word_input=pre_opt.decoder_word_input
    opt.no_connection_decoder=pre_opt.no_connection_decoder
    opt.disf_seg = pre_opt.disf_seg
    opt.no_disf_trans = pre_opt.no_disf_trans
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage)##
        if opt.train_from_pretrain:
            model_opt = opt
        else:
            model_opt = checkpoint['opt']
            model_opt.train_from_pretrain=opt.train_from_pretrain
            # I don't like reassigning attributes of opt: it's not clear
            opt.start_epoch = checkpoint['epoch'] + 1

    else:
        checkpoint = None
        model_opt = opt

    print(vars(model_opt))

    # Load fields generated from preprocess phase.
    fields = load_fields(train, valid, test, checkpoint)

    # Build model.
    model = build_model(model_opt, fields, checkpoint)

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, train, valid, test, fields, optim)


if __name__ == "__main__":
    main()
