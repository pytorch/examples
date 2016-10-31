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

# Model parameters.
parser.add_argument('-data'      , type=str, default='./data/penn', help='Location of the data corpus'               )
parser.add_argument('-checkpoint', type=str, default='./model.pt' , help='Checkpoint file path'                      )
parser.add_argument('-outf'      , type=str,   default='generated.out', help='Output file for generated text.'       )
parser.add_argument('-nwords'    , type=int, default='1000'       , help='Number of words of text to generate'       )
parser.add_argument('-seed'      , type=int,   default=1111       , help='Random seed.'                              )
parser.add_argument('-cuda'      , action='store_true'            , help='Use CUDA.'                                 )
parser.add_argument('-temperature',    type=float,   default=1.0  , help='Temperature. Higher will increase diversity')
parser.add_argument('-reportinterval', type=int,   default=100    , help='Reporting interval'                        )
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
# If the GPU is enabled, do some plumbing.

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -cuda")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

# Waiting on https://github.com/pytorch/pytorch/issues/188
# if args.cuda:
#     model.cuda()
# else:
#     model.cpu()

corpus = data.Corpus(args.data)
ntokens = corpus.dic.ntokens()

hidden = model.initHidden(1)

input = torch.LongTensor(1,1).fill_(math.floor(torch.rand(1)[0] * ntokens))
if args.cuda:
    input = input.cuda()

temperature = max(args.temperature, 1e-3)
with open(args.outf, 'w') as outf:
    for i in range(args.nwords):

        output, hidden = model(Variable(input, requires_grad=False), hidden)
        gen = torch.multinomial(output[0].data.cpu().div(temperature).exp(), 1)[0][0] # FIXME: no multinomial on GPU?
        input.fill_(gen)
        word = corpus.dic.idx2word[gen]
        outf.write(word)

        if i % 20 == 19:
            outf.write("\n")
        else:
            outf.write(" ")

        if i % args.reportinterval == 0:
            print('| Generated {}/{} words'.format(i, args.nwords))
