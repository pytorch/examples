import onmt
import onmt.utils
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

parser = argparse.ArgumentParser(description='train.lua')

parser.add_argument('-config', help="Read options from this file")

##
## **Data options**
##

parser.add_argument('-data', help="Path to the training *-train.pt file from preprocess.lua")
parser.add_argument('-save_model', help="""Model filename (the model will be saved as
                              <save_model>_epochN_PPL.pt where PPL is the validation perplexity""")
parser.add_argument('-train_from', help="If training from a checkpoint then this is the path to the pretrained model.")
parser.add_argument('-cont', action="store_true", help="If training from a checkpoint, whether to continue the training in the same configuration or not.")

##
## **Model options**
##

parser.add_argument('-layers',        type=int, default=2,   help="Number of layers in the LSTM encoder/decoder")
parser.add_argument('-rnnSize',       type=int, default=500, help="Size of LSTM hidden states")
parser.add_argument('-word_vec_size', type=int, default=500, help="Word embedding sizes")
parser.add_argument('-feat_merge', default='concat', help="Merge action for the features embeddings: concat or sum")
parser.add_argument('-feat_vec_exponent', type=int, default=0.7, help="""When using concatenation, if the feature takes N values
                                                                then the embedding dimension will be set to N^exponent""")
parser.add_argument('-feat_vec_size', type=int, default=20, help="When using sum, the common embedding size of the features")
parser.add_argument('-input_feed', type=int,    default=1,  help="Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.")
parser.add_argument('-residual',   action="store_true",     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn',       action="store_true",     help="Use a bidirectional encoder")
parser.add_argument('-brnn_merge', default='concat',        help="Merge action for the bidirectional hidden states: concat or sum")

##
## **Optimization options**
##

parser.add_argument('-max_batch_size',  type=int, default=64,  help="Maximum batch size")
parser.add_argument('-epochs',          type=int, default=13,  help="Number of training epochs")
parser.add_argument('-start_epoch',     type=int, default=0,   help="If loading from a checkpoint, the epoch from which to start")
parser.add_argument('-start_iteration', type=int, default=0,   help="If loading from a checkpoint, the iteration from which to start")
parser.add_argument('-param_init',      type=int, default=0.1, help="Parameters are initialized over uniform distribution with support (-param_init, param_init)")
parser.add_argument('-optim', default='sgd', help="Optimization method. Possible options are: sgd, adagrad, adadelta, adam")
parser.add_argument('-learning_rate', type=int, default=1, help="""Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommed settings. sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm',       type=int, default=5,   help="If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm")
parser.add_argument('-dropout',             type=int, default=0.3, help="Dropout probability. Dropout is applied between vertical LSTM stacks.")
parser.add_argument('-learning_rate_decay', type=int, default=0.5, help="""Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=8, help="Start decay after this epoch")
parser.add_argument('-curriculum', type=int, default=0, help="""For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc', help="""If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec', help="""If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.""")
# parser.add_argument('-fix_word_vecs_enc', action="store_true", help="Fix word embeddings on the encoder side")
# parser.add_argument('-fix_word_vecs_dec', action="store_true", help="Fix word embeddings on the decoder side")

##
## **Other options**
##

# GPU
parser.add_argument('-gpuid',     type=int, default=-1, help="Which gpu to use (1-indexed). < 1 = use CPU")
# parser.add_argument('-nparallel', type=int, default=1,  help="""When using GPUs, how many batches to execute in parallel.
#                             Note. this will technically change the final batch size to max_batch_size*nparallel.""")
parser.add_argument('-no_nccl', action="store_true", help="Disable usage of nccl in parallel mode.")
parser.add_argument('-disable_mem_optimization', action="store_true", help="""Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.""")

# bookkeeping
parser.add_argument('-save_every', type=int, default=0, help="""Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. """)
parser.add_argument('-report_every', type=int, default=50,   help="Print stats every this many iterations within an epoch.")
parser.add_argument('-seed',         type=int, default=3435, help="Seed for random initialization")
# parser.add_argument('-json_log', action="store_true", help="Outputs logs in JSON format.")

opt = parser.parse_args()


class NMTCriterion(nn.Container):
    def __init__(self, vocabSize, features):
        self.sub = []
        super(NMTCriterion, self).__init__()

        def makeOne(size):
            weight = torch.ones(vocabSize)
            weight[onmt.Constants.PAD] = 0
            return nn.NLLLoss(weight)

        self.sub += [makeOne(vocabSize)]
        for feature in features:
            self.sub += [makeOne(features.size())]

    def forward(self, inputs, targets):
        if len(self.sub) == 1:
            batch_size = targets.nelement()
            return self.sub[0](inputs.view(batch_size, -1), targets.view(batch_size))
        else:
            assert(False)
            loss = Variable(inputs.new(1).zero_())
            for sub, input, target in zip(self.sub, inputs, targets):
                loss += sub(input, target)
            return loss


def eval(model, criterion, data):
    loss = 0

    model.eval()
    for src, tgt in data:
        outputs = model.forward(src)
        loss = criterion.forward(outputs, tgt)

    model.train()
    return math.exp(loss / data.len)


def trainModel(model, trainData, validData, dataset):
    print(model)
    model.train()
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt']['words'].size(),
                             dataset['dicts']['tgt']['features'])

    optim = onmt.Optim(
        model.parameters(), opt.optim, opt.learning_rate,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at
    )

    # checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset)

    def trainEpoch(epoch, lastValidPpl):

        startI = opt.start_iteration

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        opt.start_iteration = 1
        ii = 1

        total_loss = 0
        for i in range(startI, len(trainData)):

            batchIdx = batchOrder[i]
            if epoch <= opt.curriculum:
                batchIdx = i

            batch = trainData[batchIdx]

            model.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch[1])
            total_loss += loss.data[0]
            loss.backward()

            # update the parameters
            optim.step(model.parameters(), opt.max_grad_norm)

            if ii % opt.report_every == 0:
                print("Done %d batches; avg loss: %g" %
                      (ii, total_loss / opt.report_every))
                total_loss = 0

            # if opt.save_every > 0 and ii % opt.save_every == 0:
            #     checkpoint.saveIteration(ii, epochState, batchOrder, not opt.json_log)

            ii += 1
        return epochState

    validPpl = 0
    for epoch in range(opt.start_epoch, opt.epochs):
        print('')
        epochState = trainEpoch(epoch, validPpl)
        validPpl = eval(model, criterion, validData)
        print('Validation perplexity. ' + validPpl)
        if opt.optim == 'sgd':
            optim.updateLearningRate(validPpl, epoch)

        # checkpoint.saveEpoch(validPpl, epochState, not opt.json_log)


def main():
    onmt.utils.Opt.initConfig(opt)
    # onmt.utils.Cuda.init(opt)

    checkpoint = {}
    # if opt.train_from is not None:
    #     assert os.path.exists(opt.train_from), 'checkpoint path invalid'
    #
    #     if not opt.json_log:
    #       print('Loading checkpoint \'' + opt.train_from + '\'...')
    #
    #     checkpoint = torch.load(opt.train_from)
    #
    #     opt.layers = checkpoint.options.layers
    #     opt.rnn_size = checkpoint.options.rnn_size
    #     opt.brnn = checkpoint.options.brnn
    #     opt.brnn_merge = checkpoint.options.brnn_merge
    #     opt.input_feed = checkpoint.options.input_feed
    #
    #     # Resume training from checkpoint
    #     if opt.cont:
    #         opt.optim = checkpoint.options.optim
    #         opt.learning_rate_decay = checkpoint.options.learning_rate_decay
    #         opt.start_decay_at = checkpoint.options.start_decay_at
    #         opt.epochs = checkpoint.options.epochs
    #         opt.curriculum = checkpoint.options.curriculum
    #
    #         opt.learning_rate = checkpoint.info.learning_rate
    #         opt.optim_states = checkpoint.info.optim_states
    #         opt.start_epoch = checkpoint.info.epoch
    #         opt.start_iteration = checkpoint.info.iteration
    #     print('Resuming training from epoch %d at iteration %d...' % (opt.start_epoch, opt.start_iteration))

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = onmt.Dataset(dataset['train']['src'], dataset['train']['tgt'], opt.max_batch_size)
    validData = onmt.Dataset(dataset['valid']['src'], dataset['valid']['tgt'], opt.max_batch_size)

    print(' * vocabulary size. source = %d; target = %d' %
            (dataset['dicts']['src']['words'].size(), dataset['dicts']['tgt']['words'].size()))
    print(' * additional features. source = %d; target = %d' %
            (len(dataset['dicts']['src']['features']), len(dataset['dicts']['tgt']['features'])))
    # print(' * maximum sequence length. source = %d; target = %d' %
    #                     (trainData.maxSourceLength, trainData.maxTargetLength))
    print(' * number of training sentences. %d' % len(trainData))
    print(' * maximum batch size. %d' % opt.max_batch_size)


    print('Building model...')

    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        encoder = onmt.Models.Encoder(opt, dataset['dicts']['src'])
        decoder = onmt.Models.Decoder(opt, dataset['dicts']['tgt'])
        model = onmt.Models.Translator(encoder, decoder)

    trainModel(model, trainData, validData, dataset)

if __name__ == "__main__":
    main()
