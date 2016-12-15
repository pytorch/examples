import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Container):
    """A container module with an encoder, an RNN (one of several flavors),
    and a decoder. Runs one RNN step at a time.
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__(
            encoder = nn.sparse.Embedding(ntoken, ninp),
            rnn = nn.RNNBase(rnn_type, ninp, nhid, nlayers, bias=False),
            decoder = nn.Linear(nhid, ntoken),
        )

        # FIXME: add stdv named argument to reset_parameters
        #        (and/or to the constructors)
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
