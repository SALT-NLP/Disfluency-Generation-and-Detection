"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
from itertools import count
import torch
import torch.nn as nn
import random as rnd

import table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, only_disf_loss, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index
        self.only_disf_loss=only_disf_loss

    def forward(self, scores, tgt):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            tgt tuple (target, align)
            align (LongTensor): ``(tgt_len, batch_size)``
            target (LongTensor): ``(tgt_len, batch_size)``
            tgt_loss_mask (LongTensor): ``(tgt_len, batch_size)``
        """
        # probabilities assigned by the model to the gold targets
        align=tgt[1]
        target=tgt[0]
        tgt_loss_mask=tgt[2]
        #print(scores, target)
        #print(scores.size(), target.size())
        target = target.view(-1)
        align = align.view(-1)
        tgt_loss_mask = tgt_loss_mask.view(-1)



        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = - probs.log()  # just NLLLoss; can the module be incorporated?

        # Drop padding.
        if self.only_disf_loss:
            loss[tgt_loss_mask == 1] = 0
        else:
            loss[tgt == self.ignore_index] = 0

        '''if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:'''
        loss = loss.sum()
        return loss

class LossCompute(nn.Module):
    def __init__(self, vocab, opt, fields,unk_index=0,
                 ignore_index=-100,smooth_eps=0):
        super(LossCompute, self).__init__()
        self.criterion = {}
        self.label_weights=torch.ones(len(fields['src_label'].vocab),dtype=torch.float,requires_grad=False,device=device)
        self.label_weights[fields['src_label'].vocab.stoi[table.IO.BOD_LABEL]]=opt.disf_label_weight
        self.label_weights[fields['src_label'].vocab.stoi[table.IO.UNK_WORD]] = 0
        self.label_weights[fields['src_label'].vocab.stoi[table.IO.PAD_WORD]] = 0
        self.criterion['lay'] = nn.NLLLoss( weight=self.label_weights,
                reduction='sum', ignore_index=ignore_index)
        if opt.no_attention:
            self.criterion['tgt'] = nn.NLLLoss(
                reduction='sum', ignore_index=ignore_index)
        else:
            if opt.no_copy:
                self.criterion['tgt'] = nn.NLLLoss(
                    reduction='sum', ignore_index=ignore_index)
            else:
                self.criterion['tgt'] = CopyGeneratorLoss(len(vocab),
                    opt.copy_attn_force, opt.only_disf_loss, unk_index=unk_index,
                    ignore_index=ignore_index)

    def compute_loss(self, pred, gold):
        loss_list = []
        for loss_name in ('lay', 'tgt'):
            if loss_name not in gold:
                continue
            '''print(loss_name)
            print(pred[loss_name].size())
            print(gold[loss_name].size())'''
            loss = self.criterion[loss_name](pred[loss_name], gold[loss_name])
            loss_list.append(loss)
        # sum up the loss functions
        return loss_list, self.label_weights[gold['lay']].sum()#sum(loss_list)

class SegLossCompute(nn.Module):
    def __init__(self, vocab, opt, fields,unk_index=0,
                 ignore_index=-100,smooth_eps=0):
        super(SegLossCompute, self).__init__()
        self.criterion= nn.NLLLoss(
                reduction='sum', ignore_index=ignore_index)

    def compute_loss(self, pred, gold):
        loss = self.criterion(pred, gold)

        return loss
