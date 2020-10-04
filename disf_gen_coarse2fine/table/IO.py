# -*- coding: utf-8 -*-

import json
import random as rnd
import numpy as np
from collections import Counter, defaultdict
from itertools import chain, count
from six import string_types

import torch
import torchtext.data
import torchtext.vocab

UNK_WORD = '<unk>'
UNK = 0
PAD_WORD = '<pad>'
PAD = 1
BOS_WORD = '<bos>'
BOS = 2
EOS_WORD = '<eos>'
EOS = 3
EOD_WORD = '<eod>'
EOD = 4
IOD_WORD = '<iod>'
IOD = 5

BOD_LABEL='E'
NBOD_LABEL='N'

DISF_LABEL='I'
FLT_LABEL='O'

special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, EOD_WORD, IOD_WORD]

def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment

def read_anno(anno_path, opt):
    js_list=[]
    num_all=0
    with open(anno_path, "r", encoding='utf-8') as reader:
        i = -1
        for line in reader:
            i += 1
            if i % 4 == 0:
                num_all+=1
                js_list.append({'sent':[token for token in line.strip().split()]})
            elif i % 4 == 1:
                continue
            if i % 4 == 2:
                l = line.strip().split()
                assert (len(l) == len(js_list[-1]['sent']))
                if not opt.include_flt:
                    if l==['O']*len(l):
                        js_list.pop()
                        continue
                if l == ['I'] * len(l):
                    js_list.pop()
                    continue
                assert (len(l)>0)
                js_list[-1]['sent_tag']=l
            else:
                continue
    print(anno_path, ' all_size:', num_all, ' disf_size:',len(js_list))
    if 'gold_diversity' in opt.__dict__ and opt.gold_diversity:
        for dic in js_list:
            disfs=[]
            indisf=0
            for i in range(len(dic['sent_tag'])):
                if indisf==0 and dic['sent_tag'][i]=='I':
                    disfs.append([dic['sent'][i]])
                    indisf=1
                elif dic['sent_tag'][i]=='I':
                    disfs[-1].append(dic['sent'][i])
                elif indisf==1 and dic['sent_tag'][i]=='O':
                    indisf=0
                else:
                    pass
            dic['disf_frags']=disfs

    for dic in js_list:
        dic['fsent']=[]
        dic['fsent_tag']=[]
        for i in range(len(dic['sent_tag'])):
            dic['fsent'].append(dic['sent'][i])
            dic['fsent_tag'].append(dic['sent_tag'][i])
            if dic['sent_tag'][i]=='I' and (i==len(dic['sent_tag'])-1 or dic['sent_tag'][i+1]=='O'):
                dic['fsent'].append(EOD_WORD)
                dic['fsent_tag'].append('I')
        line = []
        assert (len(dic['fsent_tag']) > 0)
        line.append('E' if dic['fsent_tag'][0] == 'I' else 'N')
        for i in range(len(dic['fsent_tag'])):
            if dic['fsent_tag'][i] == 'O':
                if i < len(dic['fsent_tag']) - 1 and dic['fsent_tag'][i + 1] == 'I':
                    line.append('E')
                else:
                    line.append('N')
        dic['src_label']=line
        line = []
        for w, t in zip(dic['fsent'], dic['fsent_tag']):
            if t == 'O':
                line.append(w)
        dic['src']=line
    return js_list

def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))

def filter_counter(freqs, min_freq):
    cnt = Counter()
    for k, v in freqs.items():
        if (min_freq is None) or (v >= min_freq):
            cnt[k] = v
    return cnt

def merge_vocabs(vocabs, min_freq=0, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = Counter()
    for vocab in vocabs:
        merged += filter_counter(vocab.freqs, min_freq)
    return torchtext.vocab.Vocab(merged,
                                 specials=list(special_token_list),
                                 max_size=vocab_size, min_freq=min_freq)

def _dynamic_dict(example):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src_ex_vocab = torchtext.vocab.Vocab(Counter(example["src"]), specials=[UNK_WORD, PAD_WORD])
    unk_idx = src_ex_vocab.stoi[UNK_WORD]
    assert(unk_idx==UNK)
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.tensor([unk_idx]+[src_ex_vocab.stoi[w] for w in example["src"]],dtype=torch.long)
    '''if not len(example["src"])>0:
        print(example["src"])
    assert(len(example["src"])>0)
    if not max([t for t in src_map])+1==len(src_ex_vocab):
        print(example["src"])
    assert(max([t for t in src_map])+1==len(src_ex_vocab))'''
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt_loss" in example:
        mask = torch.tensor(
            [src_ex_vocab.stoi[w] for w in example["tgt_loss"]] + [unk_idx],dtype=torch.long)
        example["alignment"] = mask
    return src_ex_vocab, example

class TableDataset(torchtext.data.Dataset):

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        if 'src' in ex.__dict__:
            return -len(ex.src)
        else:
            return -len(ex.sent)

    def __init__(self, anno, fields, opt, only_generate=False, test=False, **kwargs):
        """
        Create a TranslationDataset given paths and fields.
        anno: location of annotated data
        filter_ex: False - keep all the examples for evaluation (should not have filtered examples); True - filter examples with unmatched spans;
        """
        if isinstance(anno, string_types):
            js_list = read_anno(anno, opt)
        else:
            js_list = anno

        self.opt=opt

        if opt.disf_seg:
            sent_data = self._read_annotated_file(opt, js_list, 'sent')
            sent_examples = self._construct_examples(sent_data, 'sent')

            sent_tag_data = self._read_annotated_file(opt, js_list, 'sent_tag')
            sent_tag_examples = self._construct_examples(sent_tag_data, 'sent_tag')
        else:
            opt.no_disf_trans=False

        if opt.no_disf_trans:
            assert (opt.disf_seg)
            examples = [join_dicts(*it) for it in
                        zip(sent_examples, sent_tag_examples)]
        else:
            src_data = self._read_annotated_file(opt, js_list, 'src')
            src_examples = self._construct_examples(src_data, 'src')

            if not only_generate:
                src_label_data = self._read_annotated_file(opt, js_list, 'src_label')
                src_label_examples = self._construct_examples(src_label_data, 'src_label')

                lay_index_data = self._read_annotated_file(
                    opt, js_list, 'lay_index')
                lay_index_examples = self._construct_examples(
                    lay_index_data, 'lay_index')

                tgt_mask_data = self._read_annotated_file(
                    opt, js_list, 'tgt_mask')
                tgt_mask_examples = self._construct_examples(tgt_mask_data, 'tgt_mask')

                tgt_loss_mask_data = self._read_annotated_file(
                    opt, js_list, 'tgt_loss_mask')
                tgt_loss_mask_examples = self._construct_examples(tgt_loss_mask_data, 'tgt_loss_mask')

                tgt_data = self._read_annotated_file(opt, js_list, 'tgt')
                tgt_examples = self._construct_examples(tgt_data, 'tgt')

                tgt_loss_data = self._read_annotated_file(
                    opt, js_list, 'tgt_loss')
                tgt_loss_examples = self._construct_examples(tgt_loss_data, 'tgt_loss')

                # examples: one for each src line or (src, tgt) line pair.
                if opt.disf_seg:
                    examples = [join_dicts(*it) for it in
                                zip(sent_examples, sent_tag_examples,src_examples, src_label_examples, lay_index_examples, tgt_mask_examples,
                                    tgt_loss_mask_examples, tgt_examples, tgt_loss_examples)]

                else:
                    examples = [join_dicts(*it) for it in
                            zip(src_examples, src_label_examples, lay_index_examples, tgt_mask_examples,
                                tgt_loss_mask_examples, tgt_examples, tgt_loss_examples)]
            else:
                assert(opt.disf_seg==False)
                examples = [join_dicts(*it) for it in
                            zip(src_examples)]

        # the examples should not contain None
        len_before_filter = len(examples)
        examples = list(filter(lambda x: all(
            (v is not None for k, v in x.items())), examples))
        len_after_filter = len(examples)
        num_filter = len_before_filter - len_after_filter
        if num_filter > 0:
            print('Filter #examples (with None): {} / {} = {:.2%}'.format(num_filter,
                                                                          len_before_filter,
                                                                  num_filter / len_before_filter))
        if not opt.no_disf_trans:
            self.src_vocabs = []
            for ex_dict in examples:
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict)
                self.src_vocabs.append(src_ex_vocab)

        # Peek at the first to see which fields are used.
        ex = examples[0]
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        print('If test:',test)
        self.test=test

        super(TableDataset, self).__init__(
            self.construct_final(examples,fields,keys), fields, self.filter_pred if not test else None)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def construct_final(self,examples,fields,keys):
        exs=[]
        for i, ex in enumerate(examples):
            exs.append(torchtext.data.Example.fromlist(
                [ex[k] for k in keys] + [i],
                fields))
        return exs

    def filter_pred(self,example):
        if self.test:
            return True
        if not self.opt.no_disf_trans and (len(example.src)>self.opt.src_seq_length or len(example.tgt)>self.opt.tgt_seq_length):
            return False
        if self.opt.disf_seg and len(example.sent)>self.opt.tgt_seq_length:
            return False
        return True

    def _read_annotated_file(self, opt, js_list, field):
        l=[]
        if field == 'src':
            for dic in js_list:
                l.append(dic['src'])
            return l
        elif field == 'sent':
            for dic in js_list:
                l.append(dic['sent'])
            return l
        elif field == 'sent_tag':
            for dic in js_list:
                l.append([FLT_LABEL]+dic['sent_tag'])
            return l
        elif field == 'src_label':
            for dic in js_list:
                l.append(dic['src_label'])
            return l
        elif field=="lay_index":
            for dic in js_list:
                line = [0]
                i=1
                for w, t in zip(dic['fsent'], dic['fsent_tag']):
                    if t == 'O':
                        line.append(i)
                        i += 1
                    else:
                        line.append(0)
                l.append(line)
            return l
        elif field=="tgt_mask":
            for dic in js_list:
                if 'no_connection_decoder' in opt.__dict__ and opt.no_connection_decoder:
                    line=[1]+[1]*len(dic['fsent_tag'])
                else:
                    if 'decoder_word_input' in opt.__dict__ and opt.decoder_word_input:
                        line = []
                        line.append(0 if dic['fsent_tag'][0] == 'I' else 1)
                        for i in range(len(dic['fsent_tag'])):
                            if dic['fsent_tag'][i] == 'O' and i < len(dic['fsent_tag']) - 1 and dic['fsent_tag'][
                                i + 1] == 'I':
                                line.append(0)
                            else:
                                line.append(1)
                    else:
                        line = [0] + [1 if t == 'I' else 0 for t in dic['fsent_tag']]
                l.append(line)
            return l
        elif field=="tgt_loss_mask":
            for dic in js_list:
                line = [0 if t=='I' else 1 for t in dic['fsent_tag']] + [1]
                l.append(line)
            return l
        elif field=="tgt":
            for dic in js_list:
                l.append(dic['fsent'])
                '''line=[w if t=='I' else PAD_WORD for w, t in zip(dic['fsent'], dic['fsent_tag']) ]
                l.append(line)'''
            return l
        elif field=="tgt_loss":
            for dic in js_list:
                l.append(dic['fsent'])
            return l
        else:
            raise NotImplementedError


    def _construct_examples(self, lines, side):
        l=[]
        for words in lines:
            example_dict = {side: words}
            l.append(example_dict)
        return l

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = TableDataset.get_fields()
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def get_fields(opt=None):
        fields = {}
        fields["sent"] = torchtext.data.Field(
            init_token=IOD_WORD, pad_token=PAD_WORD, include_lengths=True, lower=opt.lower if opt else True)
        fields["sent_tag"] = torchtext.data.Field(
            pad_token=PAD_WORD, lower=False)
        fields["src"] = torchtext.data.Field(
            init_token=BOS_WORD,pad_token=PAD_WORD, include_lengths=True,lower=opt.lower if opt else True)
        fields["src_label"] = torchtext.data.Field(
            pad_token=PAD_WORD, lower=False)
        fields["lay_index"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["tgt_mask"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float, pad_token=1)
        fields["tgt_loss_mask"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long, pad_token=1)
        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, pad_token=PAD_WORD,lower=opt.lower if opt else True)

        fields["tgt_loss"] = torchtext.data.Field(
            eos_token=EOS_WORD, pad_token=PAD_WORD,lower=opt.lower if opt else True)

        fields["src_map"] = torchtext.data.Field(use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_ex_vocab"] = torchtext.data.RawField()
        fields["alignment"] = torchtext.data.Field(use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, sequential=False)

        return fields

    @staticmethod
    def build_vocab(train, dev, test, opt):
        fields = train.fields

        if opt.disf_seg:
            for field_name in ('sent', 'sent_tag'):
                fields[field_name].build_vocab(
                    train, min_freq=opt.src_words_min_frequency)

        if not opt.no_disf_trans:
            src_vocab_all = []
            # build vocabulary only based on the training set
            # the last one should be the variable 'train'
            for split in (dev, test, train,):
                fields['src'].build_vocab(split, min_freq=0)
                src_vocab_all.extend(list(fields['src'].vocab.stoi.keys()))

            # build vocabulary only based on the training set
            for field_name in ('src', 'src_label'):
                fields[field_name].build_vocab(
                    train, min_freq=opt.src_words_min_frequency)
            if opt.disf_seg:
                src_merge_name_list = ['src', 'sent']
                src_merge = merge_vocabs([fields[field_name].vocab for field_name in src_merge_name_list],
                                        min_freq=opt.src_words_min_frequency)
                for field_name in src_merge_name_list:
                    fields[field_name].vocab = src_merge

            # build vocabulary only based on the training set
            for field_name in ('tgt', 'tgt_loss'):
                fields[field_name].build_vocab(
                    train, min_freq=opt.tgt_words_min_frequency)

            tgt_merge_name_list = ['tgt', 'tgt_loss']
            tgt_merge = merge_vocabs([fields[field_name].vocab for field_name in tgt_merge_name_list],
                                     min_freq=opt.tgt_words_min_frequency)
            for field_name in tgt_merge_name_list:
                fields[field_name].vocab = tgt_merge

class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler,sort_within_batch=True) ##unordered
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

