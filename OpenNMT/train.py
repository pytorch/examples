import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from',
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=8,
                    help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
# parser.add_argument('-seed', type=int, default=3435,
#                     help="Seed for random initialization")

opt = parser.parse_args()
opt.cuda = len(opt.gpus)

print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -cuda")

if opt.cuda:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval).contiguous()

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets.contiguous(), opt.max_generator_batches)
    for out_t, targ_t in zip(outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        pred_t = generator(out_t)
        loss_t = crit(pred_t, targ_t.view(-1))
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0

    model.eval()
    for i in range(len(data)):
        batch = [x.transpose(0, 1) for x in data[i]] # must be batch first for gather/scatter in DataParallel
        outputs = model(batch)  # FIXME volatile
        targets = batch[1][:, 1:]  # exclude <s> from targets
        loss, _ = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words


def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    model.train()
    if optim.last_ppl is None:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()
    def trainEpoch(epoch):

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i
            batch = trainData[batchIdx]
            batch = [x.transpose(0, 1) for x in batch] # must be batch first for gather/scatter in DataParallel

            model.zero_grad()
            outputs = model(batch)
            targets = batch[1][:, 1:]  # exclude <s> from targets
            loss, gradOutput = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # update the parameters
            grad_norm = optim.step()

            report_loss += loss
            total_loss += loss
            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            total_words += num_words
            report_words += num_words
            if i % opt.log_interval == 0 and i > 0:
                print("Epoch %2d, %5d/%5d batches; perplexity: %6.2f; %3.0f tokens/s; %6.0f s elapsed" %
                      (epoch, i, len(trainData),
                      math.exp(report_loss / report_words),
                      report_words/(time.time()-start),
                      time.time()-start_time))

                report_loss = report_words = 0
                start = time.time()

        return total_loss / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))

        #  (2) evaluate on the validation set
        valid_loss = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)

        #  (3) maybe update the learning rate
        if opt.optim == 'sgd':
            optim.updateLearningRate(valid_loss, epoch)

        #  (4) drop a checkpoint
        checkpoint = {
            'model': model,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim,
        }
        torch.save(checkpoint,
                   '%s_e%d_%.2f.pt' % (opt.save_model, epoch, valid_ppl))


def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.train_from is None:
        encoder = onmt.Models.Encoder(opt, dicts['src'])
        decoder = onmt.Models.Decoder(opt, dicts['tgt'])
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
            nn.LogSoftmax())
        if opt.cuda > 1:
            generator = nn.DataParallel(generator, device_ids=opt.gpus)
        model = onmt.Models.NMTModel(encoder, decoder, generator)
        if opt.cuda > 1:
            model = nn.DataParallel(model, device_ids=opt.gpus)
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()

        model.generator = generator

        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optim = onmt.Optim(
            model.parameters(), opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        model = checkpoint['model']
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()
        optim = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch'] + 1

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
