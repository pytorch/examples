import sys
from tqdm import tqdm
import logging

import torch
from torch.optim import lr_scheduler
from torchtext import datasets, data

from model import SequenceLabelingModel
from args import args

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_data_iter():
    WORD = data.Field(init_token='<bos>', eos_token='<eos>', include_lengths=True)
    UD_TAG = data.Field(init_token='<bos>', eos_token='<eos>')
    PTB_TAG = data.Field(init_token='<bos>', eos_token='<eos>')
    CHAR_NESTING = data.Field(tokenize=list, init_token='<bos>', eos_token='<eos>')
    CHAR = data.NestedField(CHAR_NESTING, init_token='<bos>', eos_token='<eos>', include_lengths=True)

    train, val, test = datasets.UDPOS.splits(
        fields=((('word', 'char'), (WORD, CHAR)), ('tag', UD_TAG), ('ptbtag', PTB_TAG)),
        root='.data',
        train='en-ud-tag.v2.train.txt',
        validation='en-ud-tag.v2.dev.txt',
        test='en-ud-tag.v2.test.txt'
    )

    WORD.build_vocab(train, min_freq=args.word_min_freq)
    UD_TAG.build_vocab(train)
    PTB_TAG.build_vocab(train)
    CHAR.build_vocab(train)

    args.word2idx = WORD.vocab.stoi
    args.tag2idx = PTB_TAG.vocab.stoi
    args.char2idx = CHAR.vocab.stoi
    args.tag_bos = PTB_TAG.init_token
    args.tag_eos = PTB_TAG.eos_token
    args.tag_pad = PTB_TAG.pad_token

    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                                                                 batch_sizes=(args.train_batch_size, args.val_batch_size, args.val_batch_size),
                                                                 device=args.device, repeat=False)

    return train_iter, val_iter, test_iter


def get_optimizer_scheduler(model):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    nesterov=args.optimizer == 'nesterov')
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception('Unknown optimizer specified')
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=args.min_lr, factor=args.lr_shrink,
                                               patience=args.lr_shrink_patience, mode='max')
    return optimizer, scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStoppingCriterion(object):
    def __init__(self, patience):
        self.count = 0
        self.patience = patience

    def step(self, improved):
        self.count = 0 if improved else self.count + 1
        return self.count <= self.patience


def evaluate(model, data_iter, split):
    model.eval()
    total, acc = 0, 0
    data_iter.init_epoch()
    for batch in data_iter:
        predictions = model.decode(batch)
        for i in range(batch.batch_size):
            total += batch.word[1][i].item()
            for j in range(batch.word[1][i]):
                acc += (predictions[j, i] == batch.tag[j, i]).item()
    acc = float(acc) / total
    logger.info('%s acc: %.8f' % (split, acc))
    return acc


def train():
    train_iter, val_iter, test_iter = get_data_iter()
    model = SequenceLabelingModel(args, logger).cuda()
    optimizer, scheduler = get_optimizer_scheduler(model)
    early_stopping_criterion = EarlyStoppingCriterion(patience=args.early_stopping_patience)

    logger.info('Start training')

    for epoch in range(args.max_epochs):
        cur_lr = get_lr(optimizer)
        logger.info('Epoch %d, lr %.6f' % (epoch, cur_lr))

        model.train()
        train_score = []
        batch_num = len(train_iter)
        cur_num = 0
        train_iter.init_epoch()
        progress = tqdm(train_iter, mininterval=2, leave=False, file=sys.stdout)
        for i, batch in enumerate(progress):
            optimizer.zero_grad()

            batch_score = model.forward(batch)
            train_score.append(batch_score.item())
            cur_num += batch.batch_size
            progress.set_description(desc='%d/%d, train loss %.4f' % (i, batch_num, sum(train_score) / cur_num))
            batch_score.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_score = evaluate(model, val_iter, 'val')
        test_score = evaluate(model, test_iter, 'test')
        if not early_stopping_criterion.step(val_score):
            break
        scheduler.step(val_score)


if __name__ == '__main__':
    train()
