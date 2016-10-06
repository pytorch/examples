###############################################################################
# Language Modeling on Penn Tree Bank
###############################################################################

import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from rnn_modules import *
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Data parameters
parser.add_argument('-data'      , type=str, default='./data/penn', help='Location of the data corpus'              )
# Model parameters.
parser.add_argument('-model'     , type=str, default='LSTM'       , help='Type of recurrent net. RNN, LSTM, or GRU.')
parser.add_argument('-emsize'    , type=int, default=200          , help='Size of word embeddings'                  )
parser.add_argument('-nhid'      , type=int, default=200          , help='Number of hidden units per layer.'        )
parser.add_argument('-nlayers'   , type=int, default=2            , help='Number of layers.'                        )
# Optimization parameters.
parser.add_argument('-lr'        , type=float, default=1          , help='Initial learning rate.'                   )
parser.add_argument('-clip'      , type=float, default=0.5        , help='Gradient clipping.'                       )
parser.add_argument('-maxepoch'  , type=int,   default=6          , help='Upper epoch limit.'                       )
parser.add_argument('-batchsize' , type=int,   default=20         , help='Batch size.'                              )
parser.add_argument('-bptt'      , type=int,   default=20         , help='Sequence length.'                         )
# Device parameters.
parser.add_argument('-seed'      , type=int,   default=1111       , help='Random seed.'                             )
parser.add_argument('-cuda'      , action='store_true'            , help='Use CUDA.'                                )
# Misc parameters.
parser.add_argument('-reportint' , type=int,   default=200        , help='Report interval.'                         )
parser.add_argument('-save'      , type=str,   default='model.pt' , help='Path to save the final model.'            )
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
# If the GPU is enabled, do some plumbing.

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -cuda")

###############################################################################
## LOAD DATA
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz, bptt):
    nbatch = int(math.floor(data.size(0) / bsz / bptt))
    data = data.narrow(0, 0, nbatch * bptt * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

train = batchify(corpus.train, args.batchsize, args.bptt)
valid = batchify(corpus.valid, 10, 1)
test  = batchify(corpus.test,  10, 1)

bptt  = args.bptt
bsz   = args.batchsize

###############################################################################
# MAKE MODEL
###############################################################################

class RNNModel(nn.Container):
    """A container module with an encoder, an RNN (one of several flavors),
    and a decoder. Runs one RNN step at a time.
    """

    @staticmethod
    def name2module(name):
        if name == 'RNN':
            return RNN
        elif name == 'LSTM':
            return LSTM
        elif name == 'GRU':
            return GRU
        else:
            error("Unknown RNN module: " + name)

    def __init__(self, rnnType, ntoken, ninp, nhid, nlayers):
        rnnModule = RNNModel.name2module(rnnType)
        super(RNNModel, self).__init__(
            encoder = nn.sparse.Embedding(ntoken, ninp),
            rnn = StackedRNN(rnnModule, ninp, nhid, nlayers),
            decoder = nn.Linear(nhid, ntoken),
        )

        # FIXME: is this better than the standard init? probably
        # FIXME: we need better reset_parameters methods in stdlib
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, hidden, input):
        emb = self.encoder(input)
        hidden, output = self.rnn(hidden, emb)
        decoded = self.decoder(output)
        return hidden, decoded

    def initHidden(self, bsz):
        return self.rnn.initHidden(bsz)

model = RNNModel(args.model, corpus.dic.ntokens(), args.emsize, args.nhid, args.nlayers)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

########################################
# TRAINING
########################################

lr   = args.lr
clip = args.clip
reportinterval = args.reportint * args.batchsize

# Perform the forward pass only.
def evaluate(model, data, criterion):
    loss = 0
    hidden = model.initHidden(data.size(1))
    # Loop over validation data.
    for i in range(0, data.size(0) - 1):
        hidden, output = model(hidden, Variable(data[i], requires_grad=False))
        loss += criterion(output, Variable(data[i+1], requires_grad=False)).data

    return loss[0] / data.size(0)

# simple gradient clipping, using the total norm of the gradient
def clipGradient(model, clip):
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))

# Between bptt intervals, we want to maintain the hidden state data
# but don't want to backprop gradients across bptt intervals.
# So we have to rewrap the hidden state in a fresh Variable.
def repackageHidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackageHidden(v) for v in h)

# Loop over epochs.
prev_loss = None
for epoch in range(1, args.maxepoch+1):
    total_loss = 0
    epoch_start_time = time.time()
    # Start with an initial hidden state.
    hidden = model.initHidden(bsz)
    # Loop over the training data.
    loss = 0
    i = 0
    model.zero_grad()

    total_loss = 0
    start_time = epoch_start_time = time.time()
    while i < train.size(0) - 1:
        hidden, output = model(hidden, Variable(train[i], requires_grad=False))
        loss += criterion(output, Variable(train[i+1], requires_grad=False))
        i += 1

        if i % bptt == 0:
            loss.backward()

            clipped_lr = lr * clipGradient(model, args.clip)

            for p in model.parameters():
                p.data.sub_(p.grad.mul(clipped_lr))

            hidden = repackageHidden(hidden)
            model.zero_grad()
            total_loss += loss.data
            loss = 0

        if i % reportinterval == 0:
            cur_loss = total_loss[0] / reportinterval
            elapsed = time.time() - start_time
            print(
                    ('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                    + 'train loss {:5.2f} | train ppl {:8.2f}').format(
                epoch, i // bptt), train.size(0) // bptt, lr,
                elapsed * 1000 / reportinterval * bptt,
                cur_loss, math.exp(cur_loss)
            ))
            total_loss = 0
            start_time = time.time()

    val_loss = evaluate(model, valid, criterion)

    print(
        '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
    ))

    # The annealing schedule.
    if prev_loss and val_loss > prev_loss:
        lr = lr / 4

    prev_loss = val_loss

# Run on test data.
test_loss = evaluate(model, test, criterion)
print(
    '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)
))

if args.save != '' :
    with open(args.save, 'wb') as f:
        torch.save(model, f)
