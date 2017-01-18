# Not in use; it's for when Features are implemented
import torch
import torch.nn as nn
import math

class FeaturesEmbedding(nn.Module):

    def __init__(self, dicts, dimExponent, dim, merge):
        super(FeaturesEmbedding, self).__init__()

        self.merge = merge
        self.luts = []
        self.outputSize = dim if merge == 'sum' else 0
        for i, dict in dicts.enumerate():
            vocabSize = dict.size()
            if merge == 'sum':
                embSize = dim
            else:
                embSize = math.floor(math.pow(vocabSize, dimExponent))
                self.outputSize += embSize

            lut = nn.LookupTable(vocabSize, embSize)
            self.luts += [lut]
            self.add_module('lut_%d' % i, lut)

    def forward(self, input):
        embs = []
        for i in range(input.size(1)):
            embs += [self.luts[i](input.select(1, i))]

        if self.merge == 'sum':
            return sum(embs)
        else:
            return torch.cat(embs, 1)
