import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math, time
from alias_multinomial import alias_multinomial

class linear_nce(nn.Module):
    def __init__(self, idim, odim, unigram_prob):
        super(linear_nce, self).__init__()
        self.idim = idim
        self.odim = odim
        self.unigram_prob = Variable(torch.Tensor(unigram_prob), requires_grad=False).cuda()
        self.alias_multi = alias_multinomial(unigram_prob)
        self.weight = nn.Parameter(torch.Tensor(self.odim, self.idim))   # typically V x H
        self.bias = nn.Parameter(torch.Tensor(self.odim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(-math.log(self.odim)) # helps nce with Z=1

    def forward(self, input, target=None, mode='train', num_noise=25):
        '''
            input: N x H  where N is number of non-pad entries in T x B minibatch
            target: N x 1 target values
            mode: train|eval_full|eval_target
            (K = num_noise)
        '''
        if mode == 'eval_full':
            return F.linear(input, self.weight, self.bias)
        elif mode == 'eval_target':
            w = self.weight.index_select(0, target)
            b = self.bias.index_select(0, target)
            return torch.sum(torch.mul(input, w), 1).squeeze() + b
        elif mode == 'train':
            assert(input.size(0) == target.size(0))
            num_input = input.size(0)
            noise = Variable(self.alias_multi.draw(num_noise))

            w_target = self.weight.index_select(0, target)                      # N x H
            b_target = self.bias.index_select(0, target)                        # N
            w_noise = self.weight.index_select(0, noise)                        # K x H
            b_noise = self.bias.index_select(0, noise)                          # K

            pmt = torch.sum(torch.mul(input, w_target), 1) + b_target  # N x 1
            pmt = pmt.squeeze(1).exp() # N
            pnt = self.unigram_prob.index_select(0, target) # N

            tstart = time.time()
            pmn = F.linear(input, w_noise, b_noise)
            pmn = pmn.exp()
            pnn = self.unigram_prob.index_select(0, noise).unsqueeze(1).repeat(1, num_input, 1).view(num_input, -1) #  N x K

            return pmt, pnt, pmn, pnn
        else:
            raise ValueError('[linear_nce.forward] unknown mode={0}'.format(mode))
