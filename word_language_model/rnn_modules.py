###############################################################################
# Various RNN Modules
###############################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable

# FIXME: add CUDNN

class RNN(nn.Container):
   
    def __init__(self, ninp, nhid):
        super(RNN, self).__init__(
            i2h=nn.Linear(ninp, nhid, bias=False),
            h2h=nn.Linear(nhid, nhid, bias=False),
            sigmoid=nn.Sigmoid(),
        )
        self.ninp = ninp
        self.nhid = nhid

    def __call__(self, hidden, input):
        next = self.sigmoid(self.h2h(hidden) + self.i2h(input))
        return next, next

    def initHidden(self, bsz):
        return Variable(self.h2h.weight.data.new(bsz, self.nhid).zero_())


class LSTM(nn.Container):
   
    def __init__(self, ninp, nhid):
        super(LSTM, self).__init__(
            i2h=nn.Linear(ninp, 4 * nhid, bias=False),
            h2h=nn.Linear(nhid, 4 * nhid, bias=False),
            sigmoid=nn.Sigmoid(),
            tanh=nn.Tanh(),
        )
        self.ninp = ninp
        self.nhid = nhid

    def forward(self, hidden, input):
        c, h = hidden
        gates = self.h2h(h) + self.i2h(input)
        gates      = gates.view(input.size(0), 4, self.nhid).transpose(0, 1)

        ingate     = self.sigmoid(gates[0])
        cellgate   = self.tanh(gates[1])
        forgetgate = self.sigmoid(gates[2])
        outgate    = self.sigmoid(gates[3])

        nextc = (forgetgate * c) + (ingate * cellgate) 
        nexth = outgate * self.tanh(nextc)

        return (nextc, nexth), nexth

    def initHidden(self, bsz):
        return (Variable(self.h2h.weight.data.new(bsz, self.nhid).zero_()),
                Variable(self.h2h.weight.data.new(bsz, self.nhid).zero_()))


class GRU(nn.Container):
   
    def __init__(self, ninp, nhid):
        super(GRU, self).__init__(
            i2h=nn.Linear(ninp, 3 * nhid, bias=False),
            h2h=nn.Linear(nhid, 3 * nhid, bias=False),
            sigmoid=nn.Sigmoid(),
            tanh=nn.Tanh(),
        )
        self.ninp = ninp
        self.nhid = nhid

    def forward(self, hidden, input):
        gi = self.i2h(input).view(input.size(0), 3, self.nhid).transpose(0, 1)
        gh = self.h2h(hidden).view(input.size(0), 3, self.nhid).transpose(0, 1)

        resetgate  = self.sigmoid(gi[0] + gh[0])
        updategate = self.sigmoid(gi[1] + gh[1])

        output = self.tanh(gi[2] + resetgate * gh[2])
        nexth = hidden + updategate * (output - hidden)

        return nexth, output

    def initHidden(self, bsz):
        return Variable(self.h2h.weight.data.new(bsz, self.nhid).zero_())


class StackedRNN(nn.Container):
    def __init__(self, rnnClass, ninp, nhid, nlayers):
        super(StackedRNN, self).__init__()
        self.nlayers = nlayers
        self.layers = []
        for i in range(nlayers):
            ninp_layer = ninp if i == 0 else nhid
            layer = rnnClass(ninp_layer, nhid)
            self.layers += [layer]
            self.add_module('layer' + str(i), layer)

    def forward(self, hidden, input):
        output = input
        new_hidden = [None] * self.nlayers
        for i in range(self.nlayers):
            new_hidden[i], output = self.layers[i](hidden[i], output)
        return new_hidden, output

    def initHidden(self, bsz):
        return [m.initHidden(bsz) for m in self.layers]
