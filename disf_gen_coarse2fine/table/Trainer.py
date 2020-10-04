"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
"""
from __future__ import division
import os
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

import table
import table.modules
from table.Utils import argmax

def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]
        if len(src_vocab)<=2:
            continue
        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            #print('ti',ti)
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        #print('b',blank)
        #print(scores.size(),offset)
        if blank:
            blank = torch.tensor(blank).type_as(batch.indices)
            fill = torch.tensor(fill).type_as(batch.indices)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores

class Statistics(object):
    def __init__(self, lay_loss, tgt_loss, disf_loss, eval_result, weight_sum):
        self.disf_loss = disf_loss
        self.lay_loss = lay_loss
        self.tgt_loss= tgt_loss
        self.eval_result = eval_result
        self.weight_sum=weight_sum
        self.start_time = time.time()

    def update(self, stat):
        self.disf_loss += stat.disf_loss
        self.lay_loss += stat.lay_loss
        self.tgt_loss += stat.tgt_loss
        self.weight_sum += stat.weight_sum
        for k, v in stat.eval_result.items():
            if k in self.eval_result:
                v0 = self.eval_result[k][0] + v[0]
                v1 = self.eval_result[k][1] + v[1]
                self.eval_result[k] = (v0, v1)
            else:
                self.eval_result[k] = (v[0], v[1])

    def accuracy(self, return_str=False):
        d = sorted([(k, v)
                    for k, v in self.eval_result.items()], key=lambda x: x[0])
        if return_str:
            appendix=''
            if 'seg-token' in self.eval_result:
                assert (self.eval_result['p'][0] == self.eval_result['r'][0])
                p = self.eval_result['p'][0] / (self.eval_result['p'][1] if self.eval_result['p'][1]>0 else self.eval_result['p'][1]+1)
                r = self.eval_result['r'][0] / self.eval_result['r'][1]
                f = 2 * p * r / ((p + r) if p+r>0 else 1)
                appendix='P: '+str(p)+'; R: '+str(r)+'; F: '+str(f) +'; '

            return appendix+ '; '.join((('{}: {:.2%}'.format(k, v[0] / (1 if k=='p' and v[1]==0 else v[1]) ,)) for k, v in d))\
                   +'; Disf_loss: ' +(str(self.disf_loss/self.eval_result['seg-token'][1]) if 'seg-token' in self.eval_result else str(self.disf_loss))\
                   +'; Lay_loss: ' + (str(self.lay_loss/self.weight_sum) if 'tgt-token' in self.eval_result else str(self.lay_loss))+ '; Tgt_loss: ' \
                   + (str(self.tgt_loss/self.eval_result['tgt-token'][1]) if 'tgt-token' in self.eval_result else str(self.tgt_loss))
        else:
            return dict([(k, 100.0 * v[0] / v[1]) for k, v in d])

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):

        print(("Epoch %2d, %5d/%5d; %s; %.0f s elapsed") %
              (epoch, batch, n_batches, self.accuracy(True), time.time() - start))

        sys.stdout.flush()

    def log(self, split, logger, lr, step):
        pass


def count_accuracy(scores, target, mask=None, row=False):
    pred = argmax(scores)
    if mask is None:
        m_correct = pred.eq(target)
        num_all = m_correct.numel()
    elif row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum().item()
    return (m_correct, num_all)

def count_acc_rec(scores, target, disf_tag, mask):
    pred = argmax(scores)
    non_mask = mask.ne(1)

    pred_disf=pred.eq(disf_tag)
    target_disf=target.eq(disf_tag)
    correct_disf=pred_disf & target_disf
    pred_total=pred_disf.masked_select(non_mask).sum()
    target_total=target_disf.masked_select(non_mask).sum()
    correct=correct_disf.masked_select(non_mask).sum()

    return correct.item(), pred_total.item(), target_total.item(), non_mask.sum().item()


def aggregate_accuracy(r_dict, metric_name_list):
    m_list = []
    for metric_name in metric_name_list:
        m_list.append(r_dict[metric_name][0])
    agg = torch.stack(m_list, 0).prod(0, keepdim=False)
    return (agg.sum().item(), agg.numel())


def _debug_batch_content(vocab, ts_batch):
    seq_len = ts_batch.size(0)
    batch_size = ts_batch.size(1)
    for b in range(batch_size):
        tk_list = []
        for i in range(seq_len):
            tk = vocab.itos[ts_batch[i, b]]
            tk_list.append(tk)
        print(tk_list)


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, train_seg_loss, valid_seg_loss, optim):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_seg_loss = train_seg_loss
        self.valid_seg_loss = valid_seg_loss
        self.optim = optim

        if self.model.opt.moving_avg > 0:
            self.moving_avg = deepcopy(
                list(p for p in model.parameters()))
        else:
            self.moving_avg = None

        # Set model in training mode.
        self.model.train()

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, epoch, batch, criterion, fields):
        # 1. F-prop.
        q, q_len = batch.src
        #print('tgt_mask:',batch.tgt_mask)
        #assert((batch.tgt_mask==1).all())
        lay_out, tgt_out = self.model(
            q, q_len, None, None, batch.lay_index, batch.tgt_mask, batch.tgt, batch.src_map, self.model.opt.no_copy,
            self.model.opt.no_attention, self.model.opt.no_connection_encdec, is_seg=False)

        # _debug_batch_content(fields['lay'].vocab, argmax(lay_out.data))

        # 2. Compute loss.
        pred = {'lay': self._bottle(lay_out), 'tgt': tgt_out}
        #print("tgt_out:", tgt_out.size())
        gold = {}
        gold['lay'] = batch.src_label.view(-1)
        #print('tgt_loss:', batch.tgt_loss)
        if self.model.opt.no_attention:
            gold['tgt'] = batch.tgt_loss.view(-1)

        else:
            if self.model.opt.no_copy:
                gold['tgt'] = batch.tgt_loss.view(-1)

            else:
                gold['tgt'] = (batch.tgt_loss, batch.alignment, batch.tgt_loss_mask)

        loss,weight_sum = criterion.compute_loss(pred, gold)

        # 3. Get the batch statistics.
        if self.model.opt.no_attention:
            scores_data=self._unbottle(tgt_out.clone(), batch.batch_size)
        else:
            if not self.model.opt.no_copy:
                scores_data = collapse_copy_scores(
                    self._unbottle(tgt_out.clone(), batch.batch_size),
                    batch, fields['tgt_loss'].vocab, None)
            else:
                scores_data = self._unbottle(tgt_out.clone(), batch.batch_size)
        target_data = batch.tgt_loss.clone()

        if not self.model.opt.no_copy and not self.model.opt.no_attention:
            unk = table.IO.UNK
            correct_mask = (target_data == unk) & (batch.alignment != unk)
            offset_align = batch.alignment[correct_mask] + len(fields['tgt_loss'].vocab)
            target_data[correct_mask] += offset_align


        # Compute sum of perplexities for stats
        pred['tgt'] = scores_data
        pred['lay'] = lay_out
        gold['tgt'] = target_data
        gold['lay'] = batch.src_label
        #print(lay_out, batch.src_label)
        r_dict = {}
        for metric_name in ('lay','tgt'):
            p = pred[metric_name]
            g = gold[metric_name]
            r_dict[metric_name + '-token'] = count_accuracy(
                p, g, mask=g.eq(table.IO.PAD), row=False)
            r_dict[metric_name] = count_accuracy(
                p, g, mask=g.eq(table.IO.PAD), row=True)

        st = dict([(k, (v[0].sum().item(), v[1])) for k, v in r_dict.items()])
        st['all'] = aggregate_accuracy(r_dict, ('lay', 'tgt'))
        batch_stats = Statistics(loss[0].item(),loss[1].item(), 0.0, st, weight_sum.item())

        if epoch<self.model.opt.start_layout_loss or epoch>self.model.opt.stop_layout_loss:
            #print('1...')
            loss = loss[1]/r_dict['tgt-token'][1]
        else:
            #print('2...')
            loss = loss[0]/weight_sum*self.model.opt.layout_weight + loss[1]/r_dict['tgt-token'][1]
        return loss, batch_stats

    def seg_forward(self, epoch, batch, criterion, fields, return_pred=False):
        # 1. F-prop.
        s, s_len = batch.sent
        #print('tgt_mask:',batch.tgt_mask)
        lay_out = self.model(
            None, None, s, s_len, None, None, None, None, None,
            None, None, is_seg=True)

        # _debug_batch_content(fields['lay'].vocab, argmax(lay_out.data))

        # 2. Compute loss.
        pred = self._bottle(lay_out)
        gold = batch.sent_tag.view(-1)

        loss = criterion(pred, gold)

        # 3. Get the batch statistics.
        assert(fields['sent_tag'].vocab.stoi[table.IO.PAD_WORD]==table.IO.PAD)
        #print(lay_out, batch.src_label)
        r_dict = {}
        r_dict['correct'], r_dict['p_total'], r_dict['g_total'], total=count_acc_rec(
                lay_out[1:,:,:], batch.sent_tag[1:,:], fields['sent_tag'].vocab.stoi[table.IO.DISF_LABEL], batch.sent_tag[1:,:].eq(table.IO.PAD))

        st = {}
        st['p'] = (r_dict['correct'], r_dict['p_total'])
        st['r'] = (r_dict['correct'], r_dict['g_total'])
        st['seg-token']=(0, total)
        batch_stats = Statistics(0, 0, loss.item(), st, 0)
        loss_s = s_len.sum()
        loss = loss/loss_s
        assert(loss_s==total+batch.sent_tag.size(1))
        if return_pred:
            return loss, batch_stats, argmax(lay_out[1:, :, :])
        else:
            return loss, batch_stats


    def train(self, epoch, fields, report_func=None, test_iter=None):
        """ Called for each epoch to train. """
        total_stats = Statistics(0, 0, 0, {}, 0)
        report_stats = Statistics(0, 0, 0, {}, 0)

        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()

            if self.model.opt.disf_seg and not self.model.opt.no_disf_trans:
                if self.model.opt.trans_pretrain_epoch>0 and epoch<self.model.opt.trans_pretrain_epoch:
                    if self.model.opt.joint_loss:
                        loss, batch_stats = self.forward(
                            epoch, batch, self.train_loss, fields)
                        loss = loss * self.model.opt.disf_gen_weight
                        loss.backward()
                        self.optim.step()
                        # print(batch_stats)
                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)

                        self.model.zero_grad()

                        loss, batch_stats = self.seg_forward(
                            epoch, batch, self.train_seg_loss, fields)
                        loss.backward()
                        self.optim.step()
                        # print(batch_stats)
                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)
                    else:
                        loss, batch_stats = self.forward(
                            epoch, batch, self.train_loss, fields)
                        loss.backward()
                        self.optim.step()
                        # print(batch_stats)
                        total_stats.update(batch_stats)
                        report_stats.update(batch_stats)
                else:
                    loss, batch_stats = self.seg_forward(
                        epoch, batch, self.train_seg_loss, fields)
                    loss.backward()
                    self.optim.step()
                    # print(batch_stats)
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

            else:
                if self.model.opt.disf_seg:
                    loss, batch_stats = self.seg_forward(
                        epoch, batch, self.train_seg_loss, fields)
                else:
                    assert (self.model.opt.no_disf_trans == False)
                    #if epoch >1:
                        #print('in')
                    loss, batch_stats = self.forward(
                        epoch, batch, self.train_loss, fields)

                # print(batch_stats.eval_resultmas

                # _debug_batch_content(fields['lay'].vocab, batch.lay.data)

                # Update the parameters and statistics.
                
                loss.backward()
                if epoch >1:
                    '''print('before')
                    print('e',self.model.lay_encoder.embeddings.weight.grad)
                    print('l',self.model.lay_encoder.rnn.weight_ih_l0.grad)
                    
                    print('eq',self.model.q_encoder.embeddings.weight.grad)
                    print('lq',self.model.q_encoder.rnn.weight_ih_l0.grad)
                    test_stats = self.validate(epoch, fields, test_iter)
                    print('l',self.model.lay_encoder.rnn.weight_ih_l0)
                    print('Test accuracy: %s' % test_stats.accuracy(True))
                    print('l',self.model.lay_encoder.rnn.weight_ih_l0)'''
                self.optim.step()
                
                # print(batch_stats)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

            #print('r',report_stats.eval_result)
            #print('t', total_stats.eval_result)

            if report_func is not None:
                report_stats = report_func(
                    epoch, i, len(self.train_iter),
                    total_stats.start_time, self.optim.lr, report_stats)

            if self.model.opt.moving_avg > 0: ####
                decay_rate = min(self.model.opt.moving_avg,
                                 (1 + epoch) / (1.5 + epoch))
                for p, avg_p in zip(self.model.parameters(), self.moving_avg):
                    avg_p.mul_(decay_rate).add_(1.0 - decay_rate, p.data)
            

        return total_stats

    def validate(self, epoch, fields, valid_iter, writeFile=None):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics(0, 0, 0, {}, 0)
        preds=[]
        for batch in valid_iter:
            if self.model.opt.disf_seg:
                loss, batch_stats, pred = self.seg_forward(
                    epoch, batch, self.valid_seg_loss, fields, return_pred=True)
                if not writeFile is None:
                    for i in range(batch.indices.size(0)):
                        pred_results = [fields['sent_tag'].vocab.itos[tag_id] for tag_id, gold_id in
                                        zip(pred[:, i], batch.sent_tag[1:, i]) if not gold_id == table.IO.PAD]
                        pred_words = [fields['sent'].vocab.itos[tag_id] for tag_id, gold_id in
                                        zip(batch.sent[0][1:, i], batch.sent_tag[1:, i]) if not gold_id == table.IO.PAD]
                        preds.append((batch.indices[i].item(), pred_results, pred_words))
            else:
                assert(self.model.opt.no_disf_trans==False)
                loss, batch_stats = self.forward(
                    epoch, batch, self.valid_loss, fields)
            # Update statistics.
            stats.update(batch_stats)
        if not writeFile is None:
            preds.sort(key=lambda item: item[0])
            with open(writeFile,'w', encoding='utf-8') as writer:
                for item in preds:
                    assert(len(item[2])==len(item[1]))
                    writer.write(' '.join(item[2])+'\n')
                    writer.write(' '.join(['P']*len(item[1])) + '\n')
                    writer.write(' '.join(item[1]) + '\n')
                    writer.write('\n')
        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, eval_metric, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(eval_metric, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """

        model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'vocab': fields,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
            'moving_avg': self.moving_avg
        }
        #eval_result = valid_stats.accuracy()
        torch.save(checkpoint, os.path.join(
            opt.save_path, 'm_%d.pt' % (epoch)))
