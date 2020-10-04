import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import table
import table.IO
import table.ModelConstructor
import table.Models
import table.modules
from table.Utils import add_pad, argmax, topk
from table.Trainer import collapse_copy_scores
from table.ParseResult import GenResult
import random

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recover_layout_token(pred_list, vocab, max_sent_length):
    r_list = []
    for i in range(max_sent_length):
        r_list.append(vocab.itos[pred_list[i]])
    return r_list

def recover_target_token(lay_dec, disf_list, pred_list, vocab_tgt, src_ex_vocab, flt_gen, src, vocab_src, no_copy, gen_eod):
    '''for di in disf_list:
        for j in range(di.size(0)):
            if di[j,0] == table.IO.EOD:
                print('EEEEOOOODDDD')'''
    r_list = []
    disf_frag_list=[]
    t_list = []
    '''c_list=[]
    assert(len(src)==len(lay_dec))
    pred_len = len(src)
    for de in disf_list:
        pred_len += de.size(0)'''
    #print('src',src,len(src))
    #print('pred',pred_list,len(pred_list))
    #print('disf',disf_list)

    #assert (len(pred_list) == pred_len)
    #print(src)
    #print(len(pred_list))
    #output_list=[]
    for top_list in pred_list:
        w = src_ex_vocab.itos[top_list[0] - len(vocab_tgt)] \
            if top_list[0] >= len(vocab_tgt) else vocab_tgt.itos[top_list[0]]
        '''if gen_eod or not w == table.IO.EOD_WORD:
            c_list.append(w)
        output_list.append(w)'''

    if flt_gen:
        '''if no_copy:
            for top_list in pred_list:
                w=vocab_tgt.itos[top_list[0]]
                if gen_eod or not w==table.IO.EOD_WORD:
                    r_list.append(w)
        else:'''
        for top_list in pred_list:
            w=src_ex_vocab.itos[top_list[0]-len(vocab_tgt)] \
                if top_list[0]>=len(vocab_tgt) else vocab_tgt.itos[top_list[0]]
            if gen_eod or not w == table.IO.EOD_WORD:
                r_list.append(w)
        if r_list[-1] == table.IO.EOS_WORD:
            r_list = r_list[:-1]
        t_list=['O']*len(r_list)
        disf_frag_list.append(r_list)
        #raise NotImplementedError # for t_list
    else:
        cur_disf=0
        for i in range(len(src)):
            r_list.append(src[i])
            t_list.append('O')
            if lay_dec[i]==table.IO.BOD_LABEL:
                disf=disf_list[cur_disf]
                disf_frag_cur=[]
                '''if no_copy:
                    for i in range(disf.size(0)):
                        w=vocab_tgt.itos[disf[i][0]]
                        if not w == table.IO.EOD_WORD:
                            disf_frag_cur.append(w)
                            r_list.append(w)
                            t_list.append('I')
                else:'''
                for i in range(disf.size(0)):
                    w=src_ex_vocab.itos[disf[i][0]-len(vocab_tgt)] if disf[i][0]>=len(vocab_tgt) else vocab_tgt.itos[disf[i][0]]
                    #print(w)
                    if gen_eod or not w == table.IO.EOD_WORD:
                        disf_frag_cur.append(w)
                        r_list.append(w)
                        t_list.append('I')
                    else:
                        assert(disf.size(0)-1==i)

                disf_frag_list.append(disf_frag_cur)
                cur_disf+=1
            else:
                assert(lay_dec[i]==table.IO.NBOD_LABEL)
        assert (cur_disf==len(disf_list))
        assert(r_list[0] == table.IO.BOS_WORD)
        r_list = r_list[1:]
        t_list = t_list[1:]
        #print(len(c_list),len(r_list))
        '''if not len(c_list)==len(r_list)+1:
            print('\t'.join(c_list))
            print('\t'.join(r_list))
            print('\t'.join(output_list))
        assert (len(c_list) == len(r_list)+1)'''
        if r_list[-1] == table.IO.EOS_WORD:
            r_list = r_list[:-1]
            t_list=t_list[:-1]

        assert(len(t_list)==len(r_list))

    return r_list, disf_frag_list,t_list


class Translator(object):
    def __init__(self, opt, dummy_opt={}, device=None, parallel=False, gpu_id=0):
        # Add in default model arguments, possibly added since training.
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        self.fields = checkpoint['vocab']

        model_opt = checkpoint['opt']

        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]
        print(vars(model_opt))
        self.model_opt=model_opt
        self.device=device
        self.model = table.ModelConstructor.make_base_model(
            model_opt, self.fields, checkpoint, device=device)

        self.model.eval()
        '''if parallel:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu_id], output_device=gpu_id,
                                                        find_unused_parameters=True)'''

        if model_opt.moving_avg > 0:
            for p, avg_p in zip(self.model.parameters(), checkpoint['moving_avg']):
                p.copy_(avg_p)

        if opt.attn_ignore_small > 0:
            self.model.tgt_decoder.attn.ignore_small = opt.attn_ignore_small

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


    def run_tgt_decoder(self, embeddings, lay_list, lay_all, decoder, classifier, q, tgt, q_len, q_all, q_enc, max_disf_len, vocab,
                        src_map, batch, decoder_word_input,no_connection_decoder,no_connection_encdec):
        assert(len(lay_list)==1)
        lay_list=lay_list[0]
        assert (len(lay_list)==lay_all.size(0))
        batch_size = q.size(1)
        assert(batch_size==1)
        assert (vocab.stoi[table.IO.EOS_WORD]==table.IO.EOS)
        dec_list = []
        dec_disf_list = []
        dec_disf = []
        if self.model_opt.encoder_type=='transformer':
            decoder.init_state(q_all)
        else:
            decoder.attn.applyMaskBySeqBatch(q)
            if no_connection_encdec:
                dec_state = table.Models.RNNDecoderState(q_all, None)
            else:
                dec_state = decoder.init_decoder_state(q_all, q_enc)

        idx_lay = 0
        if no_connection_decoder:
            inp = embeddings(tgt[idx_lay, :].unsqueeze(0))
        else:
            if decoder_word_input and not lay_list[0] == table.IO.BOD_LABEL:
                inp = embeddings(tgt[idx_lay, :].unsqueeze(0))
            else:
                inp = lay_all[idx_lay, :, :].unsqueeze(0)
        idx_lay+=1

        idx_final=0
        for i in range(len(lay_list)):
            if not idx_lay == i + 1:
                print(idx_lay, i)
            assert (idx_lay == i + 1)
            assert(i+1==idx_lay)
            if self.model_opt.encoder_type == 'transformer':
                dec_all, attn_scores = decoder(inp, q_all, q_len, None, step=idx_final)#, no_attention=no_attention)
                idx_final+=1
            else:
                dec_all, dec_state, attn_scores, dec_rnn_output, concat_c = decoder(
                    inp, q_all, dec_state)
            if self.model_opt.encoder_type == 'transformer':
                if self.model_opt.no_copy or self.model_opt.no_attention:
                    dec_out = classifier(dec_all)
                else:
                    dec_out = classifier(dec_all, dec_all,
                                         dec_all, attn_scores, src_map)
            else:
                if self.model_opt.no_attention:
                    dec_out = classifier(dec_rnn_output, logsf=False)
                    # word_weights = output.squeeze().div(args.temperature).exp().cpu()
                else:
                    if not self.model_opt.no_copy:
                        dec_out = classifier(dec_all, dec_rnn_output, concat_c, attn_scores, src_map)
                    else:
                        dec_out = classifier(dec_all, logsf=False)

            assert(vocab.stoi[table.IO.UNK_WORD]==table.IO.UNK)

            dec_out[:, vocab.stoi[table.IO.UNK_WORD]] = -float('inf')
            dec_out[:, vocab.stoi[table.IO.EOS_WORD]] = -float('inf')
            dec_out[:, vocab.stoi[table.IO.EOD_WORD]] = -float('inf')
            if self.model_opt.no_copy or self.model_opt.no_attention:
                dec_out =self._unbottle(dec_out, batch.batch_size).div(self.opt.temperature).exp()
            else:
                dec_out[:, batch.src_ex_vocab[0].stoi[table.IO.UNK_WORD]+len(vocab)] = -float('inf')
                dec_out[:, batch.src_ex_vocab[0].stoi[table.IO.PAD_WORD] + len(vocab)] = -float('inf')
                if self.opt.no_in_sent_word:
                    dec_out[:, len(vocab):] = -float('inf')
                dec_out=collapse_copy_scores(self._unbottle(dec_out, batch.batch_size),batch, vocab, None)
            if self.opt.random_sample:
                topk_cpu = torch.multinomial(dec_out.view(-1), self.opt.sample_num).view(batch_size,-1)
            else:
                topk_cpu = topk(dec_out.view(batch_size, -1), 20)
            #print(topk_cpu)
            assert not topk_cpu[0,0]== vocab.stoi[table.IO.EOD_WORD]
            dec_list.append(topk_cpu.cpu())
            if lay_list[i]==table.IO.BOD_LABEL:
                dec_disf.append(topk_cpu.cpu())
                idx_disf=0
                inp = topk_cpu[:,0:1]
                assert not (inp.item()==table.IO.EOS)
                assert (inp.size(0) == 1 and inp.size(1) == 1)
                t_cur = inp.item()
                inp.masked_fill_(inp.ge(len(vocab)), table.IO.UNK)
                inp = embeddings(inp)

                while True:
                    if self.model_opt.encoder_type == 'transformer':
                        dec_all, attn_scores = decoder(inp, q_all, q_len, None,no_attention=self.model_opt.no_attention,step=idx_final)
                        idx_final+=1
                    else:
                        dec_all, dec_state, attn_scores, dec_rnn_output, concat_c = decoder(
                            inp, q_all, dec_state,no_attention=self.model_opt.no_attention)
                    if self.model_opt.encoder_type == 'transformer':
                        if self.model_opt.no_copy or self.model_opt.no_attention:
                            dec_out = classifier(dec_all)
                        else:
                            dec_out = classifier(dec_all, dec_all,
                                                 dec_all, attn_scores, src_map)
                    else:
                        if self.model_opt.no_attention:
                            dec_out = classifier(dec_rnn_output, logsf=False)
                        else:
                            if not self.model_opt.no_copy:
                                dec_out = classifier(dec_all, dec_rnn_output, concat_c, attn_scores, src_map)
                            else:
                                dec_out = classifier(dec_all, logsf=False)

                    assert (vocab.stoi[table.IO.UNK_WORD] == table.IO.UNK)
                    assert (vocab.stoi[table.IO.EOS_WORD] == table.IO.EOS)
                    dec_out[:, vocab.stoi[table.IO.UNK_WORD]] = -float('inf')
                    dec_out[:, vocab.stoi[table.IO.EOS_WORD]] = -float('inf')
                    if torch.rand(1).item()<self.opt.random_mask_eod or idx_disf>=max_disf_len-1 or t_cur==vocab.stoi[table.IO.EOD_WORD]:
                        dec_out[:, vocab.stoi[table.IO.EOD_WORD]] = -float('inf')

                    if self.model_opt.no_copy or self.model_opt.no_attention:
                        dec_out = self._unbottle(dec_out, batch.batch_size).div(self.opt.temperature).exp()
                    else:
                        dec_out[:, batch.src_ex_vocab[0].stoi[table.IO.UNK_WORD] + len(vocab)] = -float('inf')
                        dec_out[:, batch.src_ex_vocab[0].stoi[table.IO.PAD_WORD] + len(vocab)] = -float('inf')
                        if self.opt.no_in_sent_word:
                            dec_out[:, len(vocab):] = -float('inf')
                        dec_out = collapse_copy_scores(self._unbottle(dec_out, batch.batch_size), batch, vocab, None)
                    if self.opt.random_sample:
                        topk_cpu = torch.multinomial(dec_out.view(-1), self.opt.sample_num).view(batch_size, -1)
                    else:
                        topk_cpu = topk(dec_out.view(batch_size, -1), 20)
                    '''if topk_cpu[0, 0] == vocab.stoi[table.IO.EOD_WORD]:
                        print('one eod!!')'''
                    dec_list.append(topk_cpu.cpu())
                    idx_disf+=1
                    inp = topk_cpu[:,0:1]
                    assert(inp.size(0)==1 and inp.size(1)==1)
                    if idx_disf>=max_disf_len or t_cur==vocab.stoi[table.IO.EOD_WORD]:
                        if idx_lay < len(lay_list):
                            if no_connection_decoder:
                                inp = embeddings(tgt[idx_lay, :].unsqueeze(0))
                            else:
                                if decoder_word_input and not lay_list[idx_lay] == table.IO.BOD_LABEL:
                                    inp = embeddings(tgt[idx_lay, :].unsqueeze(0))
                                else:
                                    inp = lay_all[idx_lay, :, :].unsqueeze(0)

                            idx_lay += 1
                            '''if not idx_lay == i + 2:
                                print(idx_lay, i)
                                print(q)
                                print(tgt)
                                print(lay_list)'''
                            assert (idx_lay == i + 2)
                        dec_disf_list.append(torch.stack(dec_disf, 0))
                        pred_len = i+1
                        for de in dec_disf_list:
                            pred_len += de.size(0)

                        assert (len(dec_list) == pred_len)
                        dec_disf=[]
                        break
                    else:
                        dec_disf.append(topk_cpu.cpu())
                        t_cur = inp.item()
                        inp.masked_fill_(inp.ge(len(vocab)), table.IO.UNK)
                        inp = embeddings(inp)
            else:
                if idx_lay<len(lay_list):
                    if no_connection_decoder:
                        inp = embeddings(tgt[idx_lay, :].unsqueeze(0))
                    else:
                        if decoder_word_input and not lay_list[idx_lay] == table.IO.BOD_LABEL:
                            inp = embeddings(tgt[idx_lay, :].unsqueeze(0))
                        else:
                            inp = lay_all[idx_lay, :, :].unsqueeze(0)
                    idx_lay+=1
                    if not idx_lay == i + 2:
                        print(idx_lay, i)
                        print(q,tgt)
                    assert (idx_lay == i + 2)
        #assert(tgt.size(0))
        pred_len=q.size(0)
        for de in dec_disf_list:
            pred_len+=de.size(0)

        assert(len(dec_list)==pred_len)
        return torch.stack(dec_list, 0),dec_disf_list

    def translate(self, batch,js_list):
        assert(batch.batch_size==1)
        q, q_len = batch.src
        batch_size = q.size(1)

        # encoding
        if self.model_opt.encoder_type == 'transformer':
            q_all = self.model.q_encoder(q, lengths=q_len)
            q_enc = None
        else:
            q_enc, q_all = self.model.q_encoder(q, lengths=q_len)

        if not self.model_opt.encode_one_pass:
            if self.model_opt.encoder_type == 'transformer':
                lay_all = self.model.lay_encoder(q, lengths=q_len)
            else:
                lay_enc, lay_all = self.model.lay_encoder(q, lengths=q_len)
        else:
            if self.model_opt.encoder_type=='transformer':
                lay_all = q_all
            else:
                lay_enc, lay_all = q_enc, q_all

        lay_out = self.model.lay_classifier(lay_all)

        if self.opt.gold_layout:
            lay_dec = batch.src_label

        elif self.opt.random_layout:
            assert(lay_out.size(1)==1)
            count=random.randint(1, 3)
            sent_len=lay_out.size(0)
            allPos = range(sent_len)
            poses = random.choices(allPos, k=count)
            fill_idx=torch.tensor(poses, dtype=torch.long, requires_grad=False,device=self.device)
            lay_dec=torch.full((sent_len,1), self.fields['src_label'].vocab.stoi[table.IO.NBOD_LABEL],
                           dtype=torch.long, device=self.device, requires_grad=False)
            lay_dec[fill_idx,:]=self.fields['src_label'].vocab.stoi[table.IO.BOD_LABEL]
        else:
            lay_dec = argmax(lay_out)
        # recover layout
        lay_list = []
        for b in range(batch_size):
            lay_field = 'src_label'
            lay = recover_layout_token([lay_dec[i, b].item() for i in range(
                lay_dec.size(0))], self.fields[lay_field].vocab, lay_dec.size(0))
            lay_list.append(lay)


        # target decoding
        tgt_dec, tgt_disf_dec= self.run_tgt_decoder(self.model.tgt_embeddings, lay_list, lay_all, self.model.tgt_decoder,
                                       self.model.tgt_classifier, q, q, q_len, q_all, q_enc, self.opt.max_disf_len,
                                       self.fields['tgt_loss'].vocab, batch.src_map,batch,
                                       self.model_opt.decoder_word_input if 'decoder_word_input' in self.model_opt.__dict__ else False,
                                       self.model_opt.no_connection_decoder if 'no_connection_decoder' in self.model_opt.__dict__ else False,
                                       self.model_opt.no_connection_encdec)
        # recover target
        tgt_list = []
        disf_frag_list = []
        tgt_tags_list=[]
        for b in range(batch_size):
            src_word=[table.IO.BOS_WORD]+js_list[batch.indices[b]]['src']
            assert(len(src_word)==len(q[:,b]))
            tgt, disf_frags,tgt_tags= recover_target_token(lay_list[b],[disf[:,b,:] for disf in tgt_disf_dec],
                [tgt_dec[i, b] for i in range(tgt_dec.size(0))], self.fields['tgt_loss'].vocab,
                batch.src_ex_vocab[b], self.opt.flt_gen,src_word,self.fields['src'].vocab, self.model_opt.no_copy or self.model_opt.no_attention,self.opt.gen_eod)

            tgt_list.append(tgt)
            disf_frag_list.append(disf_frags)
            tgt_tags_list.append(tgt_tags)

        # (3) recover output
        indices = batch.indices.tolist()
        return [GenResult(idx, lay, tgt, disf_frags,tags)
                for idx, lay, tgt, disf_frags, tags in zip(indices, lay_list, tgt_list, disf_frag_list,tgt_tags_list)]
