import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.modules.activation import LogSoftmax
from torch.nn import functional as F

class nce_loss(Module):
    def __init__(self, size_average = True):
        super(nce_loss, self).__init__()
        self.size_average = size_average

    def forward(self, input):
        pmt, pnt, pmn, pnn = input
        assert(pmt.size(0) == pnt.size(0))
        assert(pmn.size(0) == pnn.size(0))
        assert(pmn.size(1) == pnn.size(1))
        assert(pmt.size(0) == pnn.size(0))

        N = pmt.size(0)  # num true
        K = pmn.size(1)  # num noise samples
        eps = 1e-8  # avoid 0 prob

        ptrue = pmt.div(pmt.add(pnt.mul(K)).add(eps))             # true sample
        pfalse = pnn.mul(K).div(pmn.add(pnn.mul(K)).add(eps))     # noise samples

        ptrue = ptrue.log()
        pfalse = pfalse.log().sum(1)
        loss = - ((ptrue+pfalse).sum(0))

        if self.size_average:
            loss /= N

        return loss

