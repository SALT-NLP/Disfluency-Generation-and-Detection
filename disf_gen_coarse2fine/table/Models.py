from __future__ import division
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import math
import torch.nn.functional as F

import table
from table.Utils import aeq, sort_for_pack, sequence_mask
from table.modules.embed_regularize import embedded_dropout
from table.modules.position_ffn import PositionwiseFeedForward
from table.modules import MultiHeadedAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''class TransformerModel(nn.Module):

    def __init__(self, embeddings, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        #encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,activation='gelu')
        self.transformer_encoder = TransformerEncoder(ninp, nhead, nhid, dropout,nlayers, activation='gelu')
        self.encoder = embeddings# nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        #self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, lengths=None):

        src = self.encoder(src) #* math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        mask = ~sequence_mask(lengths)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)#, self.src_mask)
        #output = self.decoder(output)
        return output'''

class PositionalEncoding(nn.Module):
    """
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


def _build_rnn(rnn_type, input_size, hidden_size, num_layers, dropout, weight_dropout, bidirectional=False):
    rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    if weight_dropout > 0:
        param_list = ['weight_hh_l' + str(i) for i in range(num_layers)]
        if bidirectional:
            param_list += [it + '_reverse' for it in param_list]
        rnn = table.modules.WeightDrop(rnn, param_list, dropout=weight_dropout)
    return rnn


class RNNEncoder(nn.Module):
    """ The standard RNN encoder. """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, dropout_i, lock_dropout, dropword, weight_dropout, embeddings):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.no_pack_padded_seq = False
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout_i)
        else:
            self.word_dropout = nn.Dropout(dropout_i)
        self.dropword = dropword ##drop ebd table

        # Use pytorch version when available.
        input_size = embeddings.embedding_dim
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size , num_layers, dropout, weight_dropout, bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        """
            Args:
                src (LongTensor):
                    padded sequences of sparse indices ``(src_len, batch)``
                lengths (LongTensor): length of each sequence ``(batch,)``
            Returns:
                (FloatTensor, FloatTensor):
                * final encoder state, used to initialize decoder (h_n,c_n) h_n of shape (num_layers * num_directions, batch, hidden_size)
                * memory bank for attention, ``(src_len, batch, hidden)``
        """
        if self.training and (self.dropword > 0):
            emb = embedded_dropout(
                self.embeddings, input, dropout=self.dropword)
        else:
            emb = self.embeddings(input)
        if self.word_dropout is not None:
            emb = self.word_dropout(emb)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        need_pack = (lengths is not None) and (not self.no_pack_padded_seq)
        if need_pack:
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if need_pack:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs

'''class TransformerEncoderLayer(nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation=='relu' else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, ninp, nhead, nhid, dropout,num_layers, activation='gelu', norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(
                ninp, nhead, nhid, dropout, activation)
                for i in range(num_layers)])
        #self. = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output'''

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout, activation):
        super(TransformerEncoderLayer, self).__init__()
        #self.self_attn = MultiHeadedAttention()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)

        #self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=attention_dropout, bias=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation=activation)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(src_len, batch_size, model_dim)``
            mask (LongTensor): ``(batch_size, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        #print(mask.size(),mask)
        #print('i',input_norm.size(),mask)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask, attn_type="self")#key_padding_mask=mask, need_weights=True, attn_mask=None)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    """
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings,activation='relu'):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.pos_emeddings = PositionalEncoding(d_model,dropout)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout, activation)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""

        emb = self.embeddings(src)
        emb = self.pos_emeddings(emb)

        #out = emb.contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        #mask = ~sequence_mask(lengths)
        # Run the forward pass of every layer of the tranformer.
        out = emb.transpose(0, 1).contiguous()
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return out.transpose(0, 1).contiguous()


class LayClassifier(nn.Module):
    def __init__(self,model_opt,vocab):
        super(LayClassifier, self).__init__()
        self.pad_idx=vocab.stoi[table.IO.PAD_WORD]
        self.unk_idx=vocab.stoi[table.IO.UNK_WORD]
        self.dropout=nn.Dropout(model_opt.dropout)
        if model_opt.encoder_type=='transformer':
            self.linear = nn.Linear(model_opt.transformer_dim, len(vocab))
        else:
            self.linear=nn.Linear(model_opt.decoder_input_size, len(vocab))

        self.ls=nn.LogSoftmax(dim=-1)

    def forward(self,input):
        input=self.dropout(input)
        logits=self.linear(input)
        logits[:, :, self.pad_idx] = -float('inf')
        logits[:, :, self.unk_idx] = -float('inf')
        return self.ls(logits)

class DisfClassifier(nn.Module):
    def __init__(self,model_opt,vocab):
        super(DisfClassifier, self).__init__()
        self.pad_idx=vocab.stoi[table.IO.PAD_WORD]
        self.unk_idx=vocab.stoi[table.IO.UNK_WORD]
        self.dropout=nn.Dropout(model_opt.dropout)
        if model_opt.encoder_type=='transformer':
            self.linear = nn.Linear(model_opt.transformer_dim, len(vocab))
        else:
            self.linear=nn.Linear(model_opt.decoder_input_size, len(vocab))

        self.ls=nn.LogSoftmax(dim=-1)
        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self,input):
        input=self.dropout(input)
        logits=self.linear(input)
        logits[:, :, self.pad_idx] = -float('inf')
        logits[:, :, self.unk_idx] = -float('inf')
        return self.ls(logits)



class SeqDecoder(nn.Module):
    def __init__(self, rnn_type, bidirectional_encoder,num_layers, embeddings, input_size, hidden_size,
                 attn_type, attn_hidden, dropout, dropout_i, lock_dropout, dropword, weight_dropout):
        super(SeqDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.input_size = input_size
        self.hidden_size = hidden_size
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout_i)
        else:
            self.word_dropout = nn.Dropout(dropout_i)
        self.dropword = dropword

        # Build the RNN.
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size, num_layers, dropout, weight_dropout)

        # Set up the standard attention.
        self.attn = table.modules.GlobalAttention(
            hidden_size, True, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, inp, context, state, no_attention=False):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:

            outputs (FloatTensor): (tgt_len,batch,input_dim) linear and activte concat_c
            state (RNNDecodersate)
            attn_scores(FloatTensor): (tgt_len,batch,src_len) probability
            rnn_outputs (FloatTensor): (seq_len, batch, num_directions(1) * hidden_size) an array of output of every time
                                     step from the rnn decoder.
            concat_c (FloatTensor): (tgt_len,batch,src_dim+input_dim) Concat input and context vector

        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        # END Args Check

        if self.embeddings is not None:
            if self.training and (self.dropword > 0):
                emb = embedded_dropout(
                    self.embeddings, inp, dropout=self.dropword)
            else:
                emb = self.embeddings(inp)
        else:
            emb = inp
        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        hidden, outputs, attns, rnn_output, concat_c = self._run_forward_pass(
            emb, context, state,no_attention=no_attention)

        # Update the state with the result.
        state.update_state(hidden)

        # Concatenates sequence of tensors along a new dimension.
        #outputs = torch.stack(outputs)
        #attns = torch.stack(attns)

        return outputs, state, attns, rnn_output, concat_c

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        return RNNDecoderState(context, tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))]))

    def _run_forward_pass(self, emb, context, state, no_attention=False):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            emb (FloatTensor): a sequence of input tokens tensors
                                of size (len x batch x ebd_dim).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (FloatTensor): final hidden state from the decoder.
            outputs (FloatTensor): (tgt_len,batch,input_dim) linear and activte concat_c
            attn_scores(FloatTensor): (tgt_len,batch,src_len) probability
            rnn_outputs (FloatTensor): (seq_len, batch, num_directions(1) * hidden_size) an array of output of every time
                                     step from the rnn decoder.
            concat_c (FloatTensor): (tgt_len,batch,src_dim+input_dim) Concat input and context vector

            #attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """

        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)

        outputs, attn_scores, concat_c=None, None, None

        # Calculate the attention.
        if not no_attention:
            attn_outputs, attn_scores, concat_c = self.attn(
                rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context.transpose(0, 1)                   # (contxt_len, batch, d)
            )

            outputs = attn_outputs    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attn_scores, rnn_output, concat_c


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))

class RNNDecoderState(DecoderState):
    def __init__(self, context, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (tuple or None): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if rnnstate is None:
            self.hidden = rnnstate
        else:
            if not isinstance(rnnstate, tuple):
                self.hidden = (rnnstate,)
            else:
                self.hidden = rnnstate

    @property
    def _all(self):
        return self.hidden

    def update_state(self, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    '''def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        v_list = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                  for e in self._all]
        self.hidden = tuple(v_list)'''

class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout, activation):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attention_dropout)###

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation=activation)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None, no_attention=False):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """

        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         attn_type="self")

        query = self.drop(query) + inputs

        if no_attention:
            mid=query
        else:
            query_norm = self.layer_norm_2(query)
            mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn


class TransformerDecoder(nn.Module):
    """
    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, attention_dropout, pad_idx,activation='relu'):
        super(TransformerDecoder, self).__init__()

        self.pos_embeddings = PositionalEncoding(d_model, dropout)
        # Decoder State
        self.state = {}
        self.pad_idx=pad_idx

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, activation)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def init_state(self, src):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    '''def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()'''

    def forward(self, emb, memory_bank, memory_lengths, tgt, step=None, no_attention=False):
        """
        Decode, possibly stepwise.
        Args:
            emb (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """


        if step == 0:
            self._init_cache(memory_bank)

        #tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.pos_embeddings(emb, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = None
        src_pad_mask =None
        if not no_attention:
            src_memory_bank = memory_bank.transpose(0, 1).contiguous()

            src_lens = memory_lengths
            src_pad_mask = ~sequence_mask(src_lens).unsqueeze(1)
        tgt_pad_mask = None
        if step is None:
            tgt_pad_mask = tgt.transpose(0, 1).eq(self.pad_idx).unsqueeze(1)
        '''pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]'''

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                no_attention=no_attention)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        return dec_outs, attn

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}

            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache


class TgtClassifier(nn.Module):
    def __init__(self, dropout, hidden_size, tgt_dict, pad_idx):
        super(TgtClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = tgt_dict.stoi[pad_idx]
        self.linear = nn.Linear(hidden_size, len(tgt_dict))

    def forward(self, hidden, logsf=True):
        dec_seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        # -> (targetL_ * batch_, rnn_size)
        hidden = self.dropout(hidden)
        hidden = hidden.view(dec_seq_len * batch_size, -1)
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        if logsf:
            logits = torch.log_softmax(logits, 1)
        return logits


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source. For each source sentence we have a `src_map` that maps each source word to an index in `tgt_dict` if it known, or else to an extra word. The copy generator is an extended version of the standard generator that computse three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead. taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary, computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    Args:
       hidden_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, dropout, hidden_size, context_size, tgt_dict, copy_prb,pad_idx):
        super(CopyGenerator, self).__init__()
        self.copy_prb = copy_prb
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = tgt_dict.stoi[pad_idx]
        self.linear = nn.Linear(hidden_size, len(tgt_dict))
        if copy_prb == 'hidden':
            self.linear_copy = nn.Linear(hidden_size, 1)
        elif copy_prb == 'hidden_context':
            self.linear_copy = nn.Linear(hidden_size + context_size, 1)
        else:
            raise NotImplementedError
        #self.tgt_dict = tgt_dict
        #self.ext_dict = ext_dict

    def forward(self, hidden, dec_rnn_output, concat_c, attn, src_map):
        """
        Compute a distribution over the target dictionary extended by the dynamic dictionary implied by compying source words.
        Args:
            hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
            dec_rnn_output (FloatTensor) ``()``
            attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
            src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        Retruns:
            prob (FloatTensor) (batch x tlen, input_size+extra_word) probability
        """
        dec_seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        # -> (targetL_ * batch_, rnn_size)
        hidden = hidden.view(dec_seq_len * batch_size, -1)
        dec_rnn_output = dec_rnn_output.view(dec_seq_len * batch_size, -1)
        concat_c = concat_c.view(dec_seq_len * batch_size, -1)
        # -> (targetL_ * batch_, sourceL_)
        attn = attn.view(dec_seq_len * batch_size, -1)

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        hidden = self.dropout(hidden)

        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        if self.copy_prb == 'hidden':
            p_copy = torch.sigmoid(self.linear_copy(dec_rnn_output))
        elif self.copy_prb == 'hidden_context':
            p_copy = torch.sigmoid(self.linear_copy(concat_c))
        else:
            raise NotImplementedError
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)

        return torch.cat([out_prob, copy_prob], 1)

class ParserModel(nn.Module):
    def __init__(self, q_encoder, disf_classifier, lay_encoder, lay_classifier, tgt_embeddings, tgt_decoder, tgt_classifier, model_opt,src_label_pad_idx):
        super(ParserModel, self).__init__()
        self.q_encoder = q_encoder
        self.lay_encoder = lay_encoder
        self.lay_classifier = lay_classifier
        self.tgt_embeddings = tgt_embeddings
        self.tgt_decoder = tgt_decoder
        self.disf_classifier=disf_classifier
        self.tgt_classifier = tgt_classifier
        self.opt = model_opt
        self.init_weights()
        #self.src_label_pad_idx=src_label_pad_idx

    def init_weights(self):
        initrange = 0.1
        if not self.tgt_embeddings is None:
            self.tgt_embeddings.weight.data.uniform_(-initrange, initrange)

    def run_copy_decoder(self, decoder, classifier, q, q_len, tgt, q_all, q_enc, inp, src_map, no_copy, no_attention, no_connection_encdec):
        if self.opt.encoder_type=='transformer':
            decoder.init_state(q_all)
        else:
            decoder.attn.applyMaskBySeqBatch(q)
            if no_connection_encdec:
                q_state = RNNDecoderState(q_all, None)
            else:
                q_state = decoder.init_decoder_state(q_all, q_enc)
        if self.opt.encoder_type=='transformer':
            dec_all, attn_scores = decoder(inp, q_all, q_len, tgt, no_attention=no_attention)
        else:
            dec_all, _, attn_scores, dec_rnn_output, concat_c = decoder(
                inp, q_all, q_state, no_attention=no_attention)
        if self.opt.encoder_type == 'transformer':
            if no_copy or no_attention:
                dec_out = classifier(dec_all)
            else:
                dec_out = classifier(dec_all, dec_all,
                                     dec_all, attn_scores, src_map)
        else:
            if no_attention:
                dec_out = classifier(dec_rnn_output)
            else:
                if no_copy:
                    dec_out = classifier(dec_all)
                else:
                    dec_out = classifier(dec_all, dec_rnn_output,
                                         concat_c, attn_scores, src_map)

        return dec_out, attn_scores

    def forward(self, q, q_len, s, s_len, lay_index, tgt_mask, tgt, src_map, no_copy, no_attention, no_connection_encdec, is_seg=False):
        """
            Compute a distribution over the target dictionary extended by the dynamic dictionary implied by compying source words.

            Retruns:
                lay_out (FloatTensor) (tlen, batch, src_label (3)) log_prob
                tgt_out (FloatTensor) (batch x tlen, input_size+extra_word) probability
        """
        if is_seg:
            if self.opt.encoder_type == 'transformer':
                q_all = self.q_encoder(s, lengths=s_len)
            else:
                q_enc, q_all = self.q_encoder(s, lengths=s_len)
            lay_out = self.disf_classifier(q_all)
            return lay_out
        else:
            batch_size = q.size(1)
            # encoding
            if self.opt.encoder_type == 'transformer':
                q_all = self.q_encoder(q, lengths=q_len)
                q_enc = None
            else:
                q_enc, q_all = self.q_encoder(q, lengths=q_len)

            # layout encoding
            # (lay_len, batch, lay_size)
            if not self.opt.encode_one_pass:
                if self.opt.encoder_type == 'transformer':
                    lay_all = self.lay_encoder(q, lengths=q_len)
                else:
                    lay_enc, lay_all = self.lay_encoder(q, lengths=q_len)
            else:
                if self.opt.encoder_type == 'transformer':
                    lay_all = q_all
                else:
                    lay_enc, lay_all = q_enc, q_all

            lay_out = self.lay_classifier(lay_all)

            # target decoding
            batch_index = torch.tensor(range(batch_size), dtype=torch.long).unsqueeze_(
                0).to(device).expand(lay_index.size(0), lay_index.size(1))
            # (tgt_len, batch, lay_size)
            lay_select = lay_all[lay_index, batch_index, :]
            # (tgt_len, batch, lay_size)
            # print('tgt:',tgt)
            tgt_inp_emb = self.tgt_embeddings(tgt)
            # (tgt_len, batch) -> (tgt_len, batch, lay_size)
            tgt_mask_expand = tgt_mask.unsqueeze(2).expand_as(tgt_inp_emb)
            # print('mask:',tgt_mask_expand)
            # print(tgt_mask_expand.size(),tgt_inp_emb.size(),lay_select)
            dec_inp = tgt_inp_emb.mul(tgt_mask_expand) + \
                      lay_select.mul(1 - tgt_mask_expand)

            tgt_out, __ = self.run_copy_decoder(
                self.tgt_decoder, self.tgt_classifier, q, q_len, tgt, q_all, q_enc, dec_inp, src_map, no_copy,
                no_attention, no_connection_encdec)

            return lay_out, tgt_out
