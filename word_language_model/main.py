import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchtext import data
from torchtext import datasets

import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=104123,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

TEXT = data.Field(time_series=True)
train, dev, test = datasets.LanguageModelingDataset.splits(
    os.path.join(args.data, ''), 'train.txt', 'valid.txt', 'test.txt', fields=TEXT)
TEXT.build_vocab(train)
eval_batch_size = 10
train_iter, dev_iter, test_iter = data.BPTTIterator.splits(
    (train, dev, test), batch_sizes=(args.batch_size, eval_batch_size, eval_batch_size),
    bptt_len=args.bptt, device=(0 if args.cuda else -1), repeat=False)

###############################################################################
# Build the model
###############################################################################

ntokens = len(TEXT.vocab)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    total_loss, total_len = 0, 0
    ntokens = len(TEXT.vocab)
    hidden = model.init_hidden(eval_batch_size)
    for batch in data_source:
        data, targets = batch.text, batch.target.view(-1)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        total_len += len(data)
        hidden = repackage_hidden(hidden)
    return total_loss[0] / total_len


def train():
    total_loss = 0
    start_time = time.time()
    ntokens = len(TEXT.vocab)
    hidden = model.init_hidden(args.batch_size)
    for i, batch in enumerate(train_iter):
        data, targets = batch.text, batch.target.view(-1)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        clipped_lr = lr * clip_gradient(model, args.clip)
        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)

        total_loss += loss.data

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_iter), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
prev_val_loss = None
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(dev_iter)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4
    prev_val_loss = val_loss


# Run on test data and save the model.
test_loss = evaluate(test_iter)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
if args.save != '':
    with open(args.save, 'wb') as f:
        torch.save(model, f)
