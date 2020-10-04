"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import table
import table.Models
import table.modules
from table.Models import ParserModel, RNNEncoder, SeqDecoder, CopyGenerator,LayClassifier,DisfClassifier, TgtClassifier, TransformerDecoder, TransformerEncoder#, TransformerModel

def make_embeddings(word_dict, vec_size, train_from_pretrain):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    if train_from_pretrain and not word_dict.vectors is None:
        assert (vec_size==word_dict.vectors.size(1))
        assert (num_word == word_dict.vectors.size(0))
        w_embeddings = nn.Embedding.from_pretrained(
            word_dict.vectors, padding_idx=word_padding_idx,freeze=False)
    else:
        w_embeddings = nn.Embedding(
            num_word, vec_size, padding_idx=word_padding_idx)
    return w_embeddings


def make_encoder(opt, embeddings):
    if opt.encoder_type=='transformer':
        attention_dropout=opt.transformer_dropout
        #return  TransformerModel(embeddings, opt.transformer_dim, opt.transformer_heads, opt.transformer_fw_dim, opt.transformer_layers, dropout=opt.transformer_dropout)
        return TransformerEncoder(opt.transformer_layers, opt.transformer_dim, opt.transformer_heads, opt.transformer_fw_dim, opt.transformer_dropout,
                 attention_dropout, embeddings, activation='gelu' if opt.gelu else 'relu')
    else:
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_enc, opt.weight_dropout, embeddings)


def make_layout_encoder(opt, embeddings):
    if opt.encoder_type=='transformer':
        attention_dropout=opt.transformer_dropout
        return TransformerEncoder(opt.transformer_layers, opt.transformer_dim, opt.transformer_heads, opt.transformer_fw_dim, opt.transformer_dropout,
                 attention_dropout, embeddings, activation='gelu' if opt.gelu else 'relu')
    else:
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.decoder_input_size, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_enc, opt.weight_dropout, embeddings)

def make_decoder(opt, fields, embeddings, input_size, no_copy, no_attention):
    if opt.encoder_type=='transformer':
        assert(fields['tgt_loss'].vocab.stoi[table.IO.PAD_WORD]==fields['tgt'].vocab.stoi[table.IO.PAD_WORD])
        attention_dropout = opt.transformer_dropout
        decoder = TransformerDecoder(opt.transformer_layers, opt.transformer_dim, opt.transformer_heads, opt.transformer_fw_dim,
                 opt.transformer_dropout, attention_dropout, fields['tgt_loss'].vocab.stoi[table.IO.PAD_WORD],activation='gelu' if opt.gelu else 'relu')
    else:
        decoder = SeqDecoder(opt.rnn_type, opt.brnn, opt.dec_layers, embeddings, input_size, opt.rnn_size,
                         opt.global_attention, opt.attn_hidden, opt.dropout, opt.dropout_i, opt.lock_dropout,
                         opt.dropword_dec, opt.weight_dropout)
    if opt.encoder_type=='transformer':
        opt.rnn_size=opt.transformer_dim
    if no_attention:
        classifier = TgtClassifier(opt.dropout, opt.rnn_size, fields['tgt_loss'].vocab,
                                   table.IO.PAD_WORD)
    else:
        if no_copy:
            classifier = TgtClassifier(opt.dropout, opt.rnn_size, fields['tgt_loss'].vocab,
                                       table.IO.PAD_WORD)
        else:
            if opt.encoder_type=='transformer':
                opt.copy_prb = 'hidden'
            classifier = CopyGenerator(opt.dropout, opt.rnn_size, opt.rnn_size, fields['tgt_loss'].vocab,
                                       opt.copy_prb, table.IO.PAD_WORD)

    return decoder, classifier


def make_base_model(model_opt, fields, checkpoint=None, device=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # embedding
    if model_opt.disf_seg:
        #print('vec', fields["sent"].vocab.vectors)
        w_embeddings = make_embeddings(fields["sent"].vocab,model_opt.word_vec_size if not model_opt.encoder_type=='transformer' else model_opt.transformer_dim,
                                       model_opt.train_from_pretrain)
    else:
        w_embeddings = make_embeddings(fields["src"].vocab,
                                       model_opt.word_vec_size if not model_opt.encoder_type == 'transformer' else model_opt.transformer_dim,
                                       model_opt.train_from_pretrain)

    # Make flt encoder.
    q_encoder = make_encoder(model_opt, w_embeddings)

    if model_opt.no_disf_trans:
        assert (model_opt.disf_seg==True)
        disf_classifier = DisfClassifier(model_opt, fields['sent_tag'].vocab)
        model = ParserModel(q_encoder, disf_classifier, None,  None, None, None, None, model_opt, None)
    else:
        if model_opt.disf_seg:
            disf_classifier = DisfClassifier(model_opt, fields['sent_tag'].vocab)
        else:
            disf_classifier = None
        # Make target decoder models.
        if model_opt.no_share_emb_layout_encoder:
            lay_encoder_embeddings = make_embeddings(
                fields['src'].vocab,
                model_opt.decoder_input_size if not model_opt.encoder_type == 'transformer' else model_opt.transformer_dim,
                model_opt.train_from_pretrain)
        else:
            assert (model_opt.decoder_input_size == model_opt.word_vec_size)
            lay_encoder_embeddings = w_embeddings

        if model_opt.no_lay_encoder:
            lay_encoder = lay_encoder_embeddings
        else:
            if model_opt.seprate_encoder:
                lay_encoder = make_layout_encoder(model_opt, lay_encoder_embeddings)
            else:
                if not model_opt.encoder_type == 'transformer':
                    assert (model_opt.decoder_input_size == model_opt.rnn_size)
                lay_encoder = q_encoder
        lay_classifier = LayClassifier(model_opt, fields['src_label'].vocab)

        tgt_embeddings = make_embeddings(
            fields['tgt'].vocab,
            model_opt.decoder_input_size if not model_opt.encoder_type == 'transformer' else model_opt.transformer_dim,
            model_opt.train_from_pretrain)
        tgt_decoder, tgt_classifier = make_decoder(
            model_opt, fields, None, model_opt.decoder_input_size, model_opt.no_copy, model_opt.no_attention)

        # Make ParserModel
        model = ParserModel(q_encoder, disf_classifier, lay_encoder, lay_classifier, tgt_embeddings, tgt_decoder, tgt_classifier,
                            model_opt, fields['src_label'].vocab.stoi[table.IO.PAD_WORD])
    '''if model_opt.encoder_type=='transformer':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)'''

    if checkpoint is not None:
        print('Loading model')
        if model_opt.train_from_pretrain:
            loaded_weights = checkpoint['model']
            own_state = model.state_dict()
            #own_state.update({k.replace('module.', ''): v for k, v in loaded_weights.items() if 'conv' in k})
            own_state.update({k: v for k, v in loaded_weights.items() if not 'embeddings' in k and not 'tgt_classifier' in k})
            model.load_state_dict(own_state)
        else:
            model.load_state_dict(checkpoint['model'])

    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(model.opt)
    model.to(device)

    return model
