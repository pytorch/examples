import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Seq2SeqModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, model_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, nhead=None):
        super(Seq2SeqModel, self).__init__()

        self.model_type = model_type
        if self.model_type in ['LSTM', 'GRU']:
            self.model = getattr(nn, self.model_type)(ninp, nhid, nlayers, dropout=dropout)
        elif self.model_type == 'Transformer':
            try:
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
            except:
                raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
            if nhead is None:
                assert self.model_type != 'Transformer'
            self.src_mask=None
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.model = TransformerEncoder(encoder_layers, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.model_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.model = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.nhead = nhead
        self.ninp = ninp
        self.decoder = nn.Linear(nhid, ntoken)

        if self.model_type == 'Transformer':
            self._reset_parameters()

        else:
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

            self.model_type = model_type
            self.nhid = nhid
            self.nlayers = nlayers

    def _reset_parameters(self):
        assert self.model_type == 'Transformer'

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        assert self.model_type == 'Transformer'
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        assert self.model_type != 'Transformer'
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, has_mask=True):
        # has_mask: apply a mask on the source sequence of the attention layer.
        if self.model_type == 'Transformer':
            if has_mask:
                device = input.device
                if self.src_mask is None or self.src_mask.size(0) != len(input):
                    mask = self._generate_square_subsequent_mask(len(input)).to(device)
                    self.src_mask = mask
            else:
                self.src_mask = None

            emb = self.drop(self.encoder(input))
            output = emb * math.sqrt(self.ninp)
            output = self.pos_encoder(output)
            output = self.model(output, self.src_mask)
            output = self.decoder(output)
            return F.log_softmax(output, dim=-1), hidden

        else:
            emb = self.drop(self.encoder(input))
            output, hidden = self.model(emb, hidden)
            output = self.drop(output)
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        if self.model_type == 'Transformer':
            return None

        weight = next(self.parameters())
        if self.model_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

