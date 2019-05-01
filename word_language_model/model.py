import torch
import torch.nn as nn
import math

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class TransformerSeq2Seq(nn.Transformer):
    r"""A transformer model applied for sequence-to-sequence transform. 
        User is able to modified the attributes as needed.
    Args:
        src_vocab: the number of vocabularies in the source sequence (required). 
        tgt_vocab: the number of vocabularies in the target sequence (required). 
        d_model: the dimension of the encoder/decoder embedding models (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    Examples::
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab)
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab, nhead=16, num_encoder_layers=12)
    """

    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__(d_model=d_model, nhead=nhead, 
                                   num_encoder_layers=num_encoder_layers, 
                                   num_decoder_layers=num_decoder_layers, 
                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout) 
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model) 
        self.pos_decoder = PositionalEncoding(d_model, dropout) 

        self.generator = Generator(d_model, tgt_vocab)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout 

        self._reset_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the mask for the src sequence (optional).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the encoder output (optional).
        Shape:
            src: [source sequence length, batch size]
            tgt: [target sequence length, batch size]
            src_mask: [source sequence length, source sequence length]
            tgt_mask: [target sequence length, target sequence length]
            memory_mask: [target sequence length, source sequence length]
            Note: The maksed positions are filled with float('-inf'). 
                  Unmasked positions are filled with float(0.0). Masks ensure that the predictions 
                  for position i depend only on the information before position i.
            output: [target sequence length, batch size, tgt_vocab]
            Note: Due to the multi-head attention architecture in the transformer model, 
                  the output sequence length of a transformer is same as the input sequence
                  (i.e. target) length of the decode. 
        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        src = self.src_embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)

        if self.generator:
            output = self.generator(output)

        return output


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens 
        in the sequence. The positional encodings have the same dimension as 
        the embeddings, so that the two can be summed. Here, we use sine and cosine 
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]    
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)


# Temporarily leave Generator module here. Will be moved somewhere else.
class Generator(nn.Module):
    r"""A generator processing the output of the decoder. It convertes sequence 
        tensors from embedding to vocabs. log_softmax function is attached to
        the end.
    Args:
        d_model: the embed dim (required).
        vocab: the number of vocabularies in the target sequence (required). 
    Examples:
        >>> generator = Generator(d_model, vocab)
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the generator model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, vocab]    
        Examples:
            >>> output = generator(x)
        """

        return F.log_softmax(self.proj(x), dim=-1)

