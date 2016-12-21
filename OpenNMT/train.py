import onmt
import onmt.utils

import argparse
import os
import torch

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
parser.add_argument('-rnn_size',      type=int, default=500, help="Size of LSTM hidden states")
parser.add_argument('-word_vec_size', type=int, default=500, help="Word embedding sizes")
parser.add_argument('-feat_merge', default='concat', help="Merge action for the features embeddings: concat or sum")
parser.add_argument('-feat_vec_exponent', type=int, default=0.7, help="""When using concatenation, if the feature takes N values
                                                                then the embedding dimension will be set to N^exponent""")
parser.add_argument('-feat_vec_size', type=int, default=20, help="When using sum, the common embedding size of the features")
parser.add_argument('-input_feed', type=int,    default=1,  help="Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.")
parser.add_argument('-residual',   action="store_true",     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn',       action="store_true",     help="Use a bidirectional encoder")
parser.add_argument('-brnn_merge', default='sum',           help="Merge action for the bidirectional hidden states: concat or sum")

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
parser.add_argument('-fix_word_vecs_enc', action="store_true", help="Fix word embeddings on the encoder side")
parser.add_argument('-fix_word_vecs_dec', action="store_true", help="Fix word embeddings on the decoder side")

##
## **Other options**
##

# GPU
parser.add_argument('-gpuid',     type=int, default=-1, help="Which gpu to use (1-indexed). < 1 = use CPU")
parser.add_argument('-nparallel', type=int, default=1,  help="""When using GPUs, how many batches to execute in parallel.
                            Note. this will technically change the final batch size to max_batch_size*nparallel.""")
parser.add_argument('-no_nccl', action="store_true", help="Disable usage of nccl in parallel mode.")
parser.add_argument('-disable_mem_optimization', action="store_true", help="""Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.""")

# bookkeeping
parser.add_argument('-save_every', type=int, default=0, help="""Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. """)
parser.add_argument('-report_every', type=int, default=50,   help="Print stats every this many iterations within an epoch.")
parser.add_argument('-seed',         type=int, default=3435, help="Seed for random initialization")
parser.add_argument('-json_log', action="store_true", help="Outputs logs in JSON format.")

opt = parser.parse_args()

# pool = onmt.utils.Parallel.ThreadPool(opt.nparallel)

def initParams(model, verbose):
    numParams = 0
    params, gradParams = {}, {}

    if verbose:
        print('Initializing parameters...')

    for mod in model.values():
        p, gp = mod.getParameters()

        if opt.train_from.len() == 0:
            p.uniform(-opt.param_init, opt.param_init)

        numParams = numParams + p.size(0)
        params += [p]
        gradParams += [gp]

    if verbose:
        print(" * number of parameters. " + numParams)

    return params, gradParams


def buildCriterion(vocabSize, features):
    criterion = nn.ParallelCriterion(False)

    def addNllCriterion(size):
        # Ignores padding value.
        w = torch.ones(size)
        w[onmt.Constants.PAD] = 0

        nll = nn.ClassNLLCriterion(w)

        # Let the training code manage loss normalization.
        nll.sizeAverage = False
        criterion.add(nll)

    addNllCriterion(vocabSize)

    for feature in features:
        addNllCriterion(feature.size())

    return criterion


def eval(model, criterion, data):
    loss = 0
    total = 0

    model.encoder.evaluate()
    model.decoder.evaluate()

    for i in range(data.batchCount()):
        batch = onmt.utils.Cuda.convert(data.getBatch(i))
        encoderStates, context = model.encoder.forward(batch)
        loss = loss + model.decoder.computeLoss(batch, encoderStates, context, criterion)
        total = total + batch.targetNonZeros

    model.encoder.training()
    model.decoder.training()

    return math.exp(loss / total)


def trainModel(model, trainData, validData, dataset, info):
    params, gradParams = {}, {}

    def initParams(idx, args, state):
        # Only logs information of the first thread.
        verbose = idx == 0 and not opt.json_log
        model = state['model']

        params, gradParams = initParams(model, verbose)
        for mod in model.values():
            mod.training()

        # define criterion of each GPU
        state['criterion'] = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words.size(),
                                                                    dataset.dicts.tgt.features))

        # # optimize memory of the first clone
        # if not opt.disable_mem_optimization:
        #     batch = onmt.utils.Cuda.convert(trainData.getBatch(1))
        #     batch.totalSize = batch.size
        #     onmt.utils.Memory.optimize(model, criterion, batch, verbose)

        return idx, state['criterion'], params, gradParams

    def _endcallback(args):
        idx, thecriterion, theparams, thegradParams = args
        if idx == 0:
            criterion = thecriterion
        params[idx] = theparams
        gradParams[idx] = thegradParams

    pool.launch(None, initParams, endcallback=_endcallback)

    optim = onmt.train.Optim(
        opt.optim, opt.learning_rate,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at
    )

    checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset)

    def trainEpoch(epoch, lastValidPpl):

        startI = opt.start_iteration
        numIterations = math.ceil(trainData.batchCount() / pool.count)

        if startI > 1 and info != None:
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim.getLearningRate(), lastValidPpl, info.epochStatus)
            batchOrder = info.batchOrder
        else:
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim.getLearningRate(), lastValidPpl)
            # shuffle mini batch order
            batchOrder = torch.randperm(trainData.batchCount())

        opt.start_iteration = 1
        ii = 1

        def trainOne(idx, args, state):

            batch = args[idx]
            if batch is None:
                return idx, 0

            # send batch data to GPU
            onmt.utils.Cuda.convert(batch)
            batch.totalSize = totalSize

            optim.zeroGrad(gradParams)

            encStates, context = model['encoder'].forward(batch)
            decOutputs = model['decoder'].forward(batch, encStates, context)

            encGradStatesOut, gradContext, loss = model['decoder'].backward(batch, decOutputs, criterion)
            model['encoder'].backward(batch, encGradStatesOut, gradContext)
            return idx, loss

        for i in range(startI, trainData.batchCount(), onmt.utils.Parallel.count):
            batches = {}
            totalSize = 0

            for j in range(math.min(onmt.utils.Parallel.count, trainData.batchCount()-i+1)):
                batchIdx = batchOrder[i+j-1]
                if epoch <= opt.curriculum:
                    batchIdx = i+j-1

                table.insert(batches, trainData.getBatch(batchIdx))
                totalSize = totalSize + batches[-1].size

            losses = {}

            def _endcallback(idx, loss):
                losses[idx] = loss

            pool.launch(None, trainOne, args=batches, endcallback=_endcallback)

            # accumulate the gradients from the different parallel threads
            XXX.accGradParams(gradParams, batches)

            # update the parameters
            optim.prepareGrad(gradParams[1], opt.max_grad_norm)
            optim.updateParams(params[1], gradParams[1])

            # sync the paramaters with the different parallel threads
            XXXsyncParams(params)

            epochState.update(batches, losses)

            if ii % opt.report_every == 0:
                epochState.log(ii, opt.json_log)

            if opt.save_every > 0 and ii % opt.save_every == 0:
                checkpoint.saveIteration(ii, epochState, batchOrder, not opt.json_log)

            ii = ii + 1
        return epochState

    validPpl = 0

    for epoch in range(opt.start_epoch, opt.epochs):
        if not opt.json_log:
            print('')

        epochState = trainEpoch(epoch, validPpl)

        validPpl = eval(model, criterion, validData)

        if not opt.json_log:
            print('Validation perplexity. ' + validPpl)

        if opt.optim == 'sgd':
            optim.updateLearningRate(validPpl, epoch)

        checkpoint.saveEpoch(validPpl, epochState, not opt.json_log)


def buildModel(idx, args, state):
    checkpoint = args
    model = state['model'] = {}

    if checkpoint.models:
        model['encoder'] = onmt.Models.loadEncoder(checkpoint.models.encoder, idx > 1)
        model['decoder'] = onmt.Models.loadDecoder(checkpoint.models.decoder, idx > 1)
    else:
        verbose = idx == 1 and not opt.json_log
        model['encoder'] = onmt.Models.buildEncoder(opt, dataset.dicts.src)
        model['decoder'] = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose)

    for mod in model.values():
        onmt.utils.Cuda.convert(mod)

    return idx, model


def main():
    onmt.utils.Opt.initConfig(opt)
    onmt.utils.Cuda.init(opt)

    checkpoint = {}

    if opt.train_from is not None:
        assert os.path.exists(opt.train_from), 'checkpoint path invalid'

        if not opt.json_log:
          print('Loading checkpoint \'' + opt.train_from + '\'...')


        checkpoint = torch.load(opt.train_from)

        opt.layers = checkpoint.options.layers
        opt.rnn_size = checkpoint.options.rnn_size
        opt.brnn = checkpoint.options.brnn
        opt.brnn_merge = checkpoint.options.brnn_merge
        opt.input_feed = checkpoint.options.input_feed

        # Resume training from checkpoint
        if opt.train_from is not None and opt.cont:
            opt.optim = checkpoint.options.optim
            opt.learning_rate_decay = checkpoint.options.learning_rate_decay
            opt.start_decay_at = checkpoint.options.start_decay_at
            opt.epochs = checkpoint.options.epochs
            opt.curriculum = checkpoint.options.curriculum

            opt.learning_rate = checkpoint.info.learning_rate
            opt.optim_states = checkpoint.info.optim_states
            opt.start_epoch = checkpoint.info.epoch
            opt.start_iteration = checkpoint.info.iteration

            if not opt.json_log:
                print('Resuming training from epoch %d at iteration %d...' % (opt.start_epoch, opt.start_iteration))

    # Create the data loader class.
    if not opt.json_log:
        print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt)
    validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt)

    trainData.setBatchSize(opt.max_batch_size)
    validData.setBatchSize(opt.max_batch_size)

    if not opt.json_log:
        print(' * vocabulary size. source = %d; target = %d' %
                            (dataset.dicts.src.words.size(), dataset.dicts.tgt.words.size()))
        print(' * additional features. source = %d; target = %d' %
                            (len(dataset.dicts.src.features), len(dataset.dicts.tgt.features)))
        print(' * maximum sequence length. source = %d; target = %d' %
                            (trainData.maxSourceLength, trainData.maxTargetLength))
        print(' * number of training sentences. %d' % len(trainData.src))
        print(' * maximum batch size. %d' % opt.max_batch_size * pool.count)
    else:
        metadata = dict(
            options=opt,
            vocabSize=dict(
                source=dataset.dicts.src.words.size(),
                target=dataset.dicts.tgt.words.size()
            ),
            additionalFeatures=dict(
                source=len(dataset.dicts.src.features),
                target=len(dataset.dicts.tgt.features)
            ),
            sequenceLength=dict(
                source=trainData.maxSourceLength,
                target=trainData.maxTargetLength
            ),
            trainingSentences = len(trainData.src)
        )

        onmt.utils.Log.logJson(metadata)


    if not opt.json_log:
        print('Building model...')

    def _endcallback(idx, themodel):
        if idx == 0:
            model = themodel

    onmt.utils.Parallel.launch(None, buildModel, args=checkpoint,
            endcallback=_endcallback)

    trainModel(model, trainData, validData, dataset, checkpoint.info)

if __name__ == "__main__":
    main()
