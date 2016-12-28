# Default decoder generator. Given RNN state, produce categorical distribution.
# Simply implements $$softmax(W h + b)$$.
#

import torch.nn as nn

def Generator(rnnSize, outputSize):
    return nn.Sequential(
        nn.Linear(rnnSize, outputSize),
        nn.LogSoftmax()
    )
