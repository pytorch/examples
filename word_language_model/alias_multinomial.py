import torch
import numpy as np
 
class alias_multinomial(object):
    '''
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        K = len(probs)
        self.q = torch.zeros(K).cuda()
        self.J = torch.LongTensor([0]*K).cuda()
    
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger  = []
        for kk, prob in enumerate(probs):
            self.q[kk] = K*prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
    
        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
    
            self.J[small] = large
            self.q[large] = (self.q[large] - 1.0) + self.q[small]
    
            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.q.clamp(0,1)
        self.J.clamp(0,K-1)

    def draw(self, N):
        '''
            Draw N samples from multinomial
        '''
        K  = self.J.size(0)
    
        r = torch.LongTensor(np.random.randint(0,K, size=N)).cuda()
        q = self.q.index_select(0, r)
        j = self.J.index_select(0, r)
        b = torch.bernoulli(q)
        oq = r.mul(b.long())
        oj = j.mul((1-b).long())

        return oq + oj
 