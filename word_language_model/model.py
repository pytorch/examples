import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# run one step of an lstm, assuming premultiplied input
@torch.jit.compile(nderivs=1)
def lstm_cell(input_, hidden, w_hh, b_hh=None):
    hx, cx = hidden
    gates = input_ + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def lstm(input, hidden, w_ih, w_hh, b_ih, b_hh):
    hx, cx = hidden[0][0], hidden[1][0]
    seq_len, batch_size, input_size = input.size()
    # pre-multiply the inputs
    input_ = F.linear(input.view(-1, input_size), w_ih, b_ih).view(seq_len, batch_size, -1)
    output = []
    for i in range(0, input.size(0)):
        hx, cx = lstm_cell(input_[i], (hx, cx), w_hh, b_hh)
        output.append(hx)
    output = torch.cat(output, 0).view(input_.size(0), *output[0].size())

    # to see details about the trace, unncomment:
    # print(lstm_cell.jit_debug_info())

    return output, (hx.view(1, *hx.size()), cx.view(1, *cx.size()))


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        # The custom code is a subset of the original options here,
        # so make sure it is actually configured to use the subset.
        # Run as:
        #  python main.py --nlayers 1 --cuda --dropout 0
        assert(dropout == 0 and nlayers == 1 and rnn_type == 'LSTM')
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
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))

        # original model, using cudnn
        # output, hidden = self.rnn(emb, hidden)
        # modified model, using our compiled lstm
        output, hidden = lstm(emb, hidden, *self.rnn.all_weights[0])

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
