###############################################################################
# Language Modeling on Penn Tree Bank
#
# With the default parameters, this should achieve ~116 perplexity on the 
# test set.
###############################################################################

import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Data parameters
parser.add_argument('-data'      , type=str, default='./data/penn', help='Location of the data corpus'              )
# Model parameters.
parser.add_argument('-model'     , type=str, default='LSTM'       , help='Type of recurrent net. RNN_TANH, RNN_RELU, LSTM, or GRU.')
parser.add_argument('-emsize'    , type=int, default=200          , help='Size of word embeddings'                  )
parser.add_argument('-nhid'      , type=int, default=200          , help='Number of hidden units per layer.'        )
parser.add_argument('-nlayers'   , type=int, default=2            , help='Number of layers.'                        )
# Optimization parameters.
parser.add_argument('-lr'        , type=float, default=20          , help='Initial learning rate.'                   )
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

def batchify(data, bsz):
    nbatch = int(math.floor(data.size(0) / bsz))
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_bsz = 10
train = batchify(corpus.train, args.batchsize)
valid = batchify(corpus.valid, eval_bsz)
test  = batchify(corpus.test,  eval_bsz)
bptt  = args.bptt
bsz   = args.batchsize

###############################################################################
# MAKE MODEL
###############################################################################

class RNNModel(nn.Container):
    """A container module with an encoder, an RNN (one of several flavors),
    and a decoder. Runs one RNN step at a time.
    """

    def __init__(self, rnnType, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__(
            encoder = nn.sparse.Embedding(ntoken, ninp),
            rnn = nn.RNNBase(rnnType, ninp, nhid, nlayers, bias=False),
            decoder = nn.Linear(nhid, ntoken),
        )

        # FIXME: add stdv named argument to reset_parameters
        #        (and/or to the constructors)
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

ntokens = corpus.dic.ntokens()
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

def initHidden(model, bsz):
    weight = next(model.parameters()).data
    if args.model == 'LSTM':
        return (Variable(weight.new(args.nlayers, bsz, args.nhid).zero_()),
                Variable(weight.new(args.nlayers, bsz, args.nhid).zero_()))
    else:
        return Variable(weight.new(args.nlayers, bsz, args.nhid).zero_())


########################################
# TRAINING
########################################

lr   = args.lr
clip = args.clip
reportinterval = args.reportint


# Perform the forward pass only.
def evaluate(model, data, criterion, bsz):
    loss = 0
    hidden = initHidden(model, bsz)
    # Loop over validation data.
    for i in range(0, data.size(0) - 1, bptt):
        seq_len = min(bptt, data.size(0) - 1 - i)
        output, hidden = model(Variable(data[i:i+seq_len], requires_grad=False), hidden)
        targets = data[i+1:i+seq_len+1].view(-1)
        loss += bptt * criterion(output.view(seq_len*bsz, -1), Variable(targets, requires_grad=False)).data
        hidden = repackageHidden(hidden)

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
    hidden = initHidden(model, bsz)

    loss = 0
    i = 0
    model.zero_grad()
    total_loss = 0
    start_time = epoch_start_time = time.time()
    ntokens = corpus.dic.ntokens()
    # Loop over the training data.
    for batch, i in enumerate(range(0, train.size(0) - 1, bptt)):
        seq_len = min(bptt, train.size(0) - 1 - i)
        output, hidden = model(Variable(train[i:i+seq_len], requires_grad=False), hidden)
        targets = train[i+1:i+seq_len+1].view(-1)
        loss = criterion(output.view(-1, ntokens), Variable(targets, requires_grad=False))

        # FIXME: this is the result of a double bug
        # bug #1: you can't have dangling nodes in the graph to call backward
        # bug #2: hidden.sum() doesn't work, gives me an error in backward, which I can't reproduce in a simple way
        #  File "/data/users/alerer/pytorch/pytorch/torch/autograd/variable.py", line 82, in backward
        #    self._execution_engine.run_backward(self, gradient, retain_variables)
        #  File "/data/users/alerer/pytorch/pytorch/torch/autograd/functions/reduce.py", line 27, in backward
        #    return grad_output.new(*self.input_size).fill_(grad_output[0])
        #ValueError: fill_ recieved an invalid combination of argument types - got (torch.cuda.FloatTensor), but expected (float value)
        if args.model == 'LSTM':
            loss += 0*hidden[0].sum(0).sum(1).sum(2)
            loss += 0*hidden[1].sum(0).sum(1).sum(2)
        else:
            loss += 0*hidden.sum(0).sum(1).sum(2)

        loss.backward()

        clipped_lr = lr * clipGradient(model, args.clip)

        for p in model.parameters():
            p.data.sub_(p.grad.mul(clipped_lr))

        hidden = repackageHidden(hidden)
        model.zero_grad()
        total_loss += loss.data
        loss = 0

        if batch % reportinterval == 0 and batch > 0:
            cur_loss = total_loss[0] / reportinterval
            elapsed = time.time() - start_time
            print(
                    ('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                    + 'train loss {:5.2f} | train ppl {:8.2f}').format(
                epoch, batch, train.size(0) // bptt, lr, elapsed * 1000 / reportinterval,
                cur_loss, math.exp(cur_loss)
            ))
            total_loss = 0
            start_time = time.time()

    val_loss = evaluate(model, valid, criterion, eval_bsz)

    print(
        '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
    ))

    # The annealing schedule.
    if prev_loss and val_loss > prev_loss:
        lr = lr / 4

    prev_loss = val_loss

# Run on test data.
test_loss = evaluate(model, test, criterion, eval_bsz)
print(
    '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)
))

if args.save != '' :
    with open(args.save, 'wb') as f:
        torch.save(model, f)
