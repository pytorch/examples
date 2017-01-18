import onmt
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import time

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', help="Read options from this file")

## **Data options**

parser.add_argument('-data', help="Path to the training *-train.pt file from preprocess.lua")
parser.add_argument('-save_model', help="""Model filename (the model will be saved as
                              <save_model>_epochN_PPL.pt where PPL is the validation perplexity""")
parser.add_argument('-train_from', help="If training from a checkpoint then this is the path to the pretrained model.")

## **Model options**

parser.add_argument('-layers',        type=int, default=2,   help="Number of layers in the LSTM encoder/decoder")
parser.add_argument('-rnn_size',       type=int, default=500, help="Size of LSTM hidden states")
parser.add_argument('-word_vec_size', type=int, default=500, help="Word embedding sizes")
parser.add_argument('-feat_merge', default='concat', help="Merge action for the features embeddings: concat or sum")
parser.add_argument('-feat_vec_exponent', type=float, default=0.7, help="""When using concatenation, if the feature takes N values
                                                                then the embedding dimension will be set to N^exponent""")
parser.add_argument('-feat_vec_size', type=int, default=20, help="When using sum, the common embedding size of the features")
parser.add_argument('-input_feed', type=int,    default=1,  help="Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.")
# parser.add_argument('-residual',   action="store_true",     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn',       action="store_true",     help="Use a bidirectional encoder")
parser.add_argument('-brnn_merge', default='concat',        help="Merge action for the bidirectional hidden states: concat or sum")

## **Optimization options**

parser.add_argument('-max_batch_size',  type=int, default=64,  help="Maximum batch size")
parser.add_argument('-max_generator_batches', type=int, default=32, help="""Maximum batches of words in a sequence to run the generator on in parallel.
                                                                           Higher is faster, but uses more memory.""")
parser.add_argument('-epochs',          type=int, default=13,  help="Number of training epochs")
parser.add_argument('-start_epoch',     type=int, default=1,   help="If loading from a checkpoint, the epoch from which to start")
parser.add_argument('-start_iteration', type=int, default=0,   help="If loading from a checkpoint, the iteration from which to start")
# this gives really bad initialization; Xavier better
parser.add_argument('-param_init',      type=float, default=0.1, help="Parameters are initialized over uniform distribution with support (-param_init, param_init)")
parser.add_argument('-optim', default='sgd', help="Optimization method. Possible options are: sgd, adagrad, adadelta, adam")
parser.add_argument('-learning_rate', type=float, default=1, help="""Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommed settings. sgd = 1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm',       type=float, default=5,   help="If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm")
parser.add_argument('-dropout',             type=float, default=0.3, help="Dropout probability. Dropout is applied between vertical LSTM stacks.")
parser.add_argument('-learning_rate_decay', type=float, default=0.5, help="""Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=8, help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true", help="""For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc', help="""If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec', help="""If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.""")

## **Other options**

# GPU
parser.add_argument('-cuda', action='store_true', help="Use CUDA")

# bookkeeping
parser.add_argument('-report_every', type=int, default=50,   help="Print stats every this many iterations within an epoch.")
# parser.add_argument('-seed',         type=int, default=3435, help="Seed for random initialization")

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -cuda")

class NMTCriterion(nn.Module):
    def __init__(self, vocabSize, features):
        self.sub = []
        super(NMTCriterion, self).__init__()

        def makeOne(size):
            weight = torch.ones(vocabSize)
            weight[onmt.Constants.PAD] = 0
            crit = nn.NLLLoss(weight, size_average=False)
            if opt.cuda:
                crit.cuda()
            return crit

        self.sub += [makeOne(vocabSize)]
        for feature in features:
            self.sub += [makeOne(features.size())]

    def forward(self, inputs, targets):
        if len(self.sub) == 1:
            total_size = targets.nelement()
            loss = self.sub[0](inputs.view(total_size, -1), targets.view(total_size))
            return loss
        else:
            assert False, "FIXME: features"
            loss = Variable(inputs.new(1).zero_())
            for sub, input, target in zip(self.sub, inputs, targets):
                loss += sub(input, target)
            return loss


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for out_t, targ_t in zip(outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        pred_t = generator(out_t)
        loss_t = crit(pred_t, targ_t)
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
        batch = data[i]
        outputs = model(batch) # FIXME volatile
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _ = memoryEfficientLoss(outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words


def trainModel(model, trainData, validData, dataset):
    print(model)
    model.train()
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt']['words'].size(),
                             dataset['dicts']['tgt']['features'])

    optim = onmt.Optim(
        model.parameters(), opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at
    )

    def trainEpoch(epoch):

        startI = opt.start_iteration
        opt.start_iteration = 1

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        start = time.time()
        for i in range(startI, len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i
            batch = trainData[batchIdx]

            model.zero_grad()
            outputs = model(batch)
            targets = batch[1][1:]  # exclude <s> from targets
            loss, gradOutput = memoryEfficientLoss(outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # update the parameters
            grad_norm = optim.step()

            report_loss += loss
            total_loss += loss
            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            total_words += num_words
            report_words += num_words
            if i % opt.report_every == 0 and i > 0:
                print("Epoch %2d, %5d/%5d batches; perplexity: %6.2f; %3.0f tokens/s" %
                      (epoch, i, len(trainData), math.exp(report_loss / report_words), report_words/(time.time()-start)))
                report_loss = report_words = 0
                start = time.time()

        return total_loss / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        train_loss = trainEpoch(epoch)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))
        valid_loss = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        if opt.optim == 'sgd':
            optim.updateLearningRate(valid_loss, epoch)

        checkpoint = {
            'model': model,
            'dicts': dataset['dicts'],
            'opt': opt,
        }
        torch.save(checkpoint, 'model_e%d_%.2f.pt' % (epoch, valid_ppl))


def main():

    checkpoint = {}
    assert opt.train_from is None, "FIXME: Load from checkpoint"

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = onmt.Dataset(dataset['train']['src'], dataset['train']['tgt'], opt.max_batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'], dataset['valid']['tgt'], opt.max_batch_size, opt.cuda)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
            (dicts['src']['words'].size(), dicts['tgt']['words'].size()))
    print(' * additional features. source = %d; target = %d' %
            (len(dicts['src']['features']), len(dicts['tgt']['features'])))
    print(' * number of training sentences. %d' % len(dataset['train']['src']['words']))
    print(' * maximum batch size. %d' % opt.max_batch_size)

    print('Building model...')

    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        encoder = onmt.Models.Encoder(opt, dicts['src'])
        decoder = onmt.Models.Decoder(opt, dicts['tgt'])
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt']['words'].size()),
            nn.LogSoftmax())
        model = onmt.Models.NMTModel(encoder, decoder, generator)

    if opt.cuda:
        model.cuda()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset)


if __name__ == "__main__":
    main()
