import os
import time

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIClassifier
from util import get_args


args = get_args()

inputs = data.Field()
answers = data.Field(sequential=False)

train, val, test = datasets.SNLI.splits(inputs, answers)

if os.path.isfile(args.vocab_cache):
    inputs.build_vocab(train, lower=args.lower)
    inputs.vocab.vectors = torch.load(args.vocab_cache)
else:
    inputs.build_vocab(train, vectors=(args.data_cache, args.word_vectors, args.d_embed), lower=args.lower)
    os.makedirs(os.path.dirname(args.vocab_cache), exist_okay=True)
    torch.save(inputs.vocab.vectors, args.vocab_cache)
answers.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=args.batch_size, device=args.gpu)
print(train_iter.batch_size)
print(len(train_iter))

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
if config.bidirectional:
    config.n_cells *= 2

model = SNLIClassifier(config)
if args.word_vectors:
    model.embed.weight.data = inputs.vocab.vectors
model.cuda()
criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Val/Loss     Accuracy  Val/Accuracy'
val_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):
        model.train(); opt.zero_grad()
        iterations += 1
        answer = model(batch)
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total
        loss = criterion(answer, batch.label)
        loss.backward(); opt.step()
        if iterations % args.save_every == 0:
            torch.save(model, os.path.join(args.save_path,
                'snapshot_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)))
        if iterations % args.val_every == 0:
            model.eval(); val_iter.init_epoch()
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(val_iter):
                 answer = model(dev_batch)
                 n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                 dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(val)
            print(val_log_template.format(time.time()-start,
                epoch, iterations, batch_idx, len(train_iter),
                100. * batch_idx / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model, os.path.join(args.save_path,
                    'best_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)))
        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start,
                epoch, iterations, batch_idx, len(train_iter),
                100. * batch_idx / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))


