import onmt.init
import os

parser = argparse.ArgumentParser(description='train.lua')

cmd.option('-config', "Read options from this file")

##
## **Data options**
##

cmd.option('-data', help="Path to the training *-train.t7 file from preprocess.lua")
cmd.option('-save_model', help="""Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity""")
cmd.option('-train_from', help="If training from a checkpoint then this is the path to the pretrained model.")
cmd.option('-cont', action="store_true", help="If training from a checkpoint, whether to continue the training in the same configuration or not.")

##
## **Model options**
##

cmd.option('-layers',        type=int, default=2,   help="Number of layers in the LSTM encoder/decoder")
cmd.option('-rnn_size',      type=int, default=500, help="Size of LSTM hidden states")
cmd.option('-word_vec_size', type=int, default=500, help="Word embedding sizes")
cmd.option('-feat_merge', 'concat', "Merge action for the features embeddings: concat or sum")
cmd.option('-feat_vec_exponent', type=int, default=0.7, help="""When using concatenation, if the feature takes N values
                                                                then the embedding dimension will be set to N^exponent""")
cmd.option('-feat_vec_size', type=int, default=20, help="When using sum, the common embedding size of the features")
cmd.option('-input_feed', type=int,    default=1,  help="Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.")
cmd.option('-residual',   action="store_true",     help="Add residual connections between RNN layers.")
cmd.option('-brnn',       action="store_true",     help="Use a bidirectional encoder")
cmd.option('-brnn_merge', default='sum',           help="Merge action for the bidirectional hidden states: concat or sum")

##
## **Optimization options**
##

cmd.option('-max_batch_size',  type=int, default=64,  default="Maximum batch size")
cmd.option('-epochs',          type=int, default=13,  default="Number of training epochs")
cmd.option('-start_epoch',     type=int, default=1,   default="If loading from a checkpoint, the epoch from which to start")
cmd.option('-start_iteration', type=int, default=1,   default="If loading from a checkpoint, the iteration from which to start")
cmd.option('-param_init',      type=int, default=0.1, default="Parameters are initialized over uniform distribution with support (-param_init, param_init)")
cmd.option('-optim', default='sgd', help="Optimization method. Possible options are: sgd, adagrad, adadelta, adam")
cmd.option('-learning_rate', type=int, default=1, help="""Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommed settings. sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1""")
cmd.option('-max_grad_norm',       type=int, default=5,   help="If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm")
cmd.option('-dropout',             type=int, default=0.3, help="Dropout probability. Dropout is applied between vertical LSTM stacks.")
cmd.option('-learning_rate_decay', type=int, default=0.5, help="""Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit""")
cmd.option('-start_decay_at', 9, "Start decay after this epoch")
cmd.option('-curriculum', 0, """For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.""")
cmd.option('-pre_word_vecs_enc', '', """If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.""")
cmd.option('-pre_word_vecs_dec', '', """If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.""")
cmd.option('-fix_word_vecs_enc', action="store_true", help="Fix word embeddings on the encoder side")
cmd.option('-fix_word_vecs_dec', action="store_true", help="Fix word embeddings on the decoder side")

##
## **Other options**
##

# GPU
cmd.option('-gpuid',     type=int, default=-1, help="Which gpu to use (1-indexed). < 1 = use CPU")
cmd.option('-nparallel', type=int, default=1,  help="""When using GPUs, how many batches to execute in parallel.
                            Note. this will technically change the final batch size to max_batch_size*nparallel.""")
cmd.option('-no_nccl', action="store_true", help="Disable usage of nccl in parallel mode.")
cmd.option('-disable_mem_optimization', action="store_true", help="""Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.""")

# bookkeeping
cmd.option('-save_every', type=int, default=0, help="""Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. """)
cmd.option('-report_every', type=int, default=50,   help="Print stats every this many iterations within an epoch.")
cmd.option('-seed',         type=int, default=3435, help="Seed for random initialization")
cmd.option('-json_log', action="store_true", help="Outputs logs in JSON format.")

opt = cmd.parse(arg)

def initParams(model, verbose):
    numParams = 0
    params = {}
    gradParams = {}

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

    for j = in range(len(features)):
        addNllCriterion(features[j].size())

    return criterion


def eval(model, criterion, data):
    loss = 0
    total = 0

    model.encoder.evaluate()
    model.decoder.evaluate()

    for i = 1, data.batchCount():
        batch = onmt.utils.Cuda.convert(data.getBatch(i))
        encoderStates, context = model.encoder.forward(batch)
        loss = loss + model.decoder.computeLoss(batch, encoderStates, context, criterion)
        total = total + batch.targetNonZeros


    model.encoder.training()
    model.decoder.training()

    return math.exp(loss / total)


def trainModel(model, trainData, validData, dataset, info):
    params, gradParams = {}, {}

    def func1(idx):
        # Only logs information of the first thread.
        verbose = idx == 1 and not opt.json_log

        _G.params, _G.gradParams = initParams(_G.model, verbose)
        for mod in _G.model.values():
            mod.training()


        # define criterion of each GPU
        _G.criterion = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words.size(),
                                                            dataset.dicts.tgt.features))

        # optimize memory of the first clone
        if not opt.disable_mem_optimization:
            batch = onmt.utils.Cuda.convert(trainData.getBatch(1))
            batch.totalSize = batch.size
            onmt.utils.Memory.optimize(_G.model, _G.criterion, batch, verbose)


       return idx, _G.criterion, _G.params, _G.gradParams

    def func2(idx, thecriterion, theparams, thegradParams):
        if idx == 0:
            criterion = thecriterion
        params[idx] = theparams
        gradParams[idx] = thegradParams

    onmt.utils.Parallel.launch(None, func1, func2)

    optim = onmt.train.Optim.new({
        method = opt.optim,
        numModels = len(params[0]),
        learningRate = opt.learning_rate,
        learningRateDecay = opt.learning_rate_decay,
        startDecayAt = opt.start_decay_at,
        optimStates = opt.optim_states
    })

    checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset)

    def trainEpoch(epoch, lastValidPpl):

        startI = opt.start_iteration
        numIterations = math.ceil(trainData.batchCount() / onmt.utils.Parallel.count)

        if startI > 1 and info != None:
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim.getLearningRate(), lastValidPpl, info.epochStatus)
            batchOrder = info.batchOrder
        else:
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim.getLearningRate(), lastValidPpl)
            # shuffle mini batch order
            batchOrder = torch.randperm(trainData.batchCount())


        opt.start_iteration = 1
        ii = 1

        def trainOne(idx):
            _G.batch = batches[idx]
            if _G.batch is None:
                return idx, 0

            # s batch data to GPU
            onmt.utils.Cuda.convert(_G.batch)
            _G.batch.totalSize = totalSize

            optim.zeroGrad(_G.gradParams)

            encStates, context = _G.model.encoder.forward(_G.batch)
            decOutputs = _G.model.decoder.forward(_G.batch, encStates, context)

            encGradStatesOut, gradContext, loss = _G.model.decoder.backward(_G.batch, decOutputs, _G.criterion)
            _G.model.encoder.backward(_G.batch, encGradStatesOut, gradContext)
            return idx, loss

        for i = startI, trainData.batchCount(), onmt.utils.Parallel.count:
            batches = {}
            totalSize = 0

            for j = 1, math.min(onmt.utils.Parallel.count, trainData.batchCount()-i+1):
                batchIdx = batchOrder[i+j-1]
                if epoch <= opt.curriculum:
                    batchIdx = i+j-1

                table.insert(batches, trainData.getBatch(batchIdx))
                totalSize = totalSize + batches[-1].size


            losses = {}

            onmt.utils.Parallel.launch(None, trainOne, lambda idx, loss: losses[idx]=loss)

            # accumulate the gradients from the different parallel threads
            onmt.utils.Parallel.accGradParams(gradParams, batches)

            # update the parameters
            optim.prepareGrad(gradParams[1], opt.max_grad_norm)
            optim.updateParams(params[1], gradParams[1])

            # sync the paramaters with the different parallel threads
            onmt.utils.Parallel.syncParams(params)

            epochState.update(batches, losses)

            if ii % opt.report_every == 0:
                epochState.log(ii, opt.json_log)

            if opt.save_every > 0 and ii % opt.save_every == 0:
                checkpoint.saveIteration(ii, epochState, batchOrder, not opt.json_log)

            ii = ii + 1
        return epochState

    validPpl = 0

    for epoch = opt.start_epoch, opt.epochs:
        if not opt.json_log:
            print('')

        epochState = trainEpoch(epoch, validPpl)

        validPpl = eval(model, criterion, validData)

        if not opt.json_log:
            print('Validation perplexity. ' + validPpl)


        if opt.optim == 'sgd':
            optim.updateLearningRate(validPpl, epoch)

        checkpoint.saveEpoch(validPpl, epochState, not opt.json_log)

def buildModel(idx):
    _G.model = {}

    if checkpoint.models:
        _G.model.encoder = onmt.Models.loadEncoder(checkpoint.models.encoder, idx > 1)
        _G.model.decoder = onmt.Models.loadDecoder(checkpoint.models.decoder, idx > 1)
    else:
        verbose = idx == 1 and not opt.json_log
        _G.model.encoder = onmt.Models.buildEncoder(opt, dataset.dicts.src)
        _G.model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose)


    for _, mod in pairs(_G.model):
        onmt.utils.Cuda.convert(mod)


    return idx, _G.model

def main():
    requiredOptions = {
        "data",
        "save_model"
    }

    onmt.utils.Opt.init(opt, requiredOptions)
    onmt.utils.Cuda.init(opt)
    onmt.utils.Parallel.init(opt)

    checkpoint = {}

    if opt.train_from.len() > 0:
        assert(os.path.exists(opt.train_from), 'checkpoint path invalid')

        if not opt.json_log:
          print('Loading checkpoint \'' + opt.train_from + '\'...')


        checkpoint = torch.load(opt.train_from)

        opt.layers = checkpoint.options.layers
        opt.rnn_size = checkpoint.options.rnn_size
        opt.brnn = checkpoint.options.brnn
        opt.brnn_merge = checkpoint.options.brnn_merge
        opt.input_feed = checkpoint.options.input_feed

        # Resume training from checkpoint
        if opt.train_from.len() > 0 and opt.cont:
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
        print('Loading data from \'' + opt.data + '\'...')

    dataset = torch.load(opt.data)

    trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt)
    validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt)

    trainData.setBatchSize(opt.max_batch_size)
    validData.setBatchSize(opt.max_batch_size)

    if not opt.json_log:
        print(string.format(' * vocabulary size. source = %d; target = %d',
                            dataset.dicts.src.words.size(), dataset.dicts.tgt.words:size()))
        print(string.format(' * additional features. source = %d; target = %d',
                            len(dataset.dicts.src.features), #dataset.dicts.tgt.features))
        print(string.format(' * maximum sequence length. source = %d; target = %d',
                            trainData.maxSourceLength, trainData.maxTargetLength))
        print(string.format(' * number of training sentences. %d', len(trainData.src)))
        print(string.format(' * maximum batch size. %d', opt.max_batch_size * onmt.utils.Parallel.count))
    else:
        metadata = {
            options = opt,
            vocabSize = {
                source = dataset.dicts.src.words.size(),
                target = dataset.dicts.tgt.words.size()
            },
            additionalFeatures = {
                source = len(dataset.dicts.src.features),
                target = len(dataset.dicts.tgt.features)
            },
            sequenceLength = {
                source = trainData.maxSourceLength,
                target = trainData.maxTargetLength
            },
            trainingSentences = len(trainData.src)
        }

        onmt.utils.Log.logJson(metadata)


    if not opt.json_log:
        print('Building model...')

    onmt.utils.Parallel.launch(None, buildModel,
            lambda idx, themodel: if idx == 1: model = themodel)

    trainModel(model, trainData, validData, dataset, checkpoint.info)


main()
