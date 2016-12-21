require('onmt.init')

local path = require('pl.path')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data', '', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
cmd:option('-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                      then the embedding dimension will be set to N^exponent]])
cmd:option('-feat_vec_size', 20, [[When using sum, the common embedding size of the features]])
cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
cmd:option('-brnn', false, [[Use a bidirectional encoder]])
cmd:option('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-start_iteration', 1, [[If loading from a checkpoint, the iteration from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-learning_rate_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use (1-indexed). < 1 = use CPU]])
cmd:option('-nparallel', 1, [[When using GPUs, how many batches to execute in parallel.
                            Note: this will technically change the final batch size to max_batch_size*nparallel.]])
cmd:option('-no_nccl', false, [[Disable usage of nccl in parallel mode.]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

-- bookkeeping
cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-json_log', false, [[Outputs logs in JSON format.]])

local opt = cmd:parse(arg)

local function initParams(model, verbose)
  local numParams = 0
  local params = {}
  local gradParams = {}

  if verbose then
    print('Initializing parameters...')
  end

  for _, mod in pairs(model) do
    local p, gp = mod:getParameters()

    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
  end

  if verbose then
    print(" * number of parameters: " .. numParams)
  end

  return params, gradParams
end

local function buildCriterion(vocabSize, features)
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  addNllCriterion(vocabSize)

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  return criterion
end

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  model.encoder:evaluate()
  model.decoder:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local encoderStates, context = model.encoder:forward(batch)
    loss = loss + model.decoder:computeLoss(batch, encoderStates, context, criterion)
    total = total + batch.targetNonZeros
  end

  model.encoder:training()
  model.decoder:training()

  return math.exp(loss / total)
end

local function trainModel(model, trainData, validData, dataset, info)
  local params, gradParams = {}, {}
  local criterion

  onmt.utils.Parallel.launch(nil, function(idx)
    -- Only logs information of the first thread.
    local verbose = idx == 1 and not opt.json_log

    _G.params, _G.gradParams = initParams(_G.model, verbose)
    for _, mod in pairs(_G.model) do
      mod:training()
    end

    -- define criterion of each GPU
    _G.criterion = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words:size(),
                                                          dataset.dicts.tgt.features))

    -- optimize memory of the first clone
    if not opt.disable_mem_optimization then
      local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
      batch.totalSize = batch.size
      onmt.utils.Memory.optimize(_G.model, _G.criterion, batch, verbose)
    end

    return idx, _G.criterion, _G.params, _G.gradParams
  end, function(idx, thecriterion, theparams, thegradParams)
    if idx == 1 then criterion = thecriterion end
    params[idx] = theparams
    gradParams[idx] = thegradParams
  end)

  local optim = onmt.train.Optim.new({
    method = opt.optim,
    numModels = #params[1],
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    startDecayAt = opt.start_decay_at,
    optimStates = opt.optim_states
  })

  local checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset)

  local function trainEpoch(epoch, lastValidPpl)
    local epochState
    local batchOrder

    local startI = opt.start_iteration
    local numIterations = math.ceil(trainData:batchCount() / onmt.utils.Parallel.count)

    if startI > 1 and info ~= nil then
      epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
      batchOrder = info.batchOrder
    else
      epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
      -- shuffle mini batch order
      batchOrder = torch.randperm(trainData:batchCount())
    end

    opt.start_iteration = 1
    local iter = 1

    for i = startI, trainData:batchCount(), onmt.utils.Parallel.count do
      local batches = {}
      local totalSize = 0

      for j = 1, math.min(onmt.utils.Parallel.count, trainData:batchCount()-i+1) do
        local batchIdx = batchOrder[i+j-1]
        if epoch <= opt.curriculum then
          batchIdx = i+j-1
        end
        table.insert(batches, trainData:getBatch(batchIdx))
        totalSize = totalSize + batches[#batches].size
      end

      local losses = {}

      onmt.utils.Parallel.launch(nil, function(idx)
        _G.batch = batches[idx]
        if _G.batch == nil then
          return idx, 0
        end

        -- send batch data to GPU
        onmt.utils.Cuda.convert(_G.batch)
        _G.batch.totalSize = totalSize

        optim:zeroGrad(_G.gradParams)

        local encStates, context = _G.model.encoder:forward(_G.batch)
        local decOutputs = _G.model.decoder:forward(_G.batch, encStates, context)

        local encGradStatesOut, gradContext, loss = _G.model.decoder:backward(_G.batch, decOutputs, _G.criterion)
        _G.model.encoder:backward(_G.batch, encGradStatesOut, gradContext)
        return idx, loss
      end,
      function(idx, loss) losses[idx]=loss end)

      -- accumulate the gradients from the different parallel threads
      onmt.utils.Parallel.accGradParams(gradParams, batches)

      -- update the parameters
      optim:prepareGrad(gradParams[1], opt.max_grad_norm)
      optim:updateParams(params[1], gradParams[1])

      -- sync the paramaters with the different parallel threads
      onmt.utils.Parallel.syncParams(params)

      epochState:update(batches, losses)

      if iter % opt.report_every == 0 then
        epochState:log(iter, opt.json_log)
      end
      if opt.save_every > 0 and iter % opt.save_every == 0 then
        checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
      end
      iter = iter + 1
    end

    return epochState
  end

  local validPpl = 0

  for epoch = opt.start_epoch, opt.epochs do
    if not opt.json_log then
      print('')
    end

    local epochState = trainEpoch(epoch, validPpl)

    validPpl = eval(model, criterion, validData)

    if not opt.json_log then
      print('Validation perplexity: ' .. validPpl)
    end

    if opt.optim == 'sgd' then
      optim:updateLearningRate(validPpl, epoch)
    end

    checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
  end
end


local function main()
  local requiredOptions = {
    "data",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint = {}

  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')

    if not opt.json_log then
      print('Loading checkpoint \'' .. opt.train_from .. '\'...')
    end

    checkpoint = torch.load(opt.train_from)

    opt.layers = checkpoint.options.layers
    opt.rnn_size = checkpoint.options.rnn_size
    opt.brnn = checkpoint.options.brnn
    opt.brnn_merge = checkpoint.options.brnn_merge
    opt.input_feed = checkpoint.options.input_feed

    -- Resume training from checkpoint
    if opt.train_from:len() > 0 and opt.continue then
      opt.optim = checkpoint.options.optim
      opt.learning_rate_decay = checkpoint.options.learning_rate_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.epochs = checkpoint.options.epochs
      opt.curriculum = checkpoint.options.curriculum

      opt.learning_rate = checkpoint.info.learning_rate
      opt.optim_states = checkpoint.info.optim_states
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      if not opt.json_log then
        print('Resuming training from epoch ' .. opt.start_epoch
                .. ' at iteration ' .. opt.start_iteration .. '...')
      end
    end
  end

  -- Create the data loader class.
  if not opt.json_log then
    print('Loading data from \'' .. opt.data .. '\'...')
  end

  local dataset = torch.load(opt.data)

  local trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt)
  local validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  if not opt.json_log then
    print(string.format(' * vocabulary size: source = %d; target = %d',
                        dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size()))
    print(string.format(' * additional features: source = %d; target = %d',
                        #dataset.dicts.src.features, #dataset.dicts.tgt.features))
    print(string.format(' * maximum sequence length: source = %d; target = %d',
                        trainData.maxSourceLength, trainData.maxTargetLength))
    print(string.format(' * number of training sentences: %d', #trainData.src))
    print(string.format(' * maximum batch size: %d', opt.max_batch_size * onmt.utils.Parallel.count))
  else
    local metadata = {
      options = opt,
      vocabSize = {
        source = dataset.dicts.src.words:size(),
        target = dataset.dicts.tgt.words:size()
      },
      additionalFeatures = {
        source = #dataset.dicts.src.features,
        target = #dataset.dicts.tgt.features
      },
      sequenceLength = {
        source = trainData.maxSourceLength,
        target = trainData.maxTargetLength
      },
      trainingSentences = #trainData.src
    }

    onmt.utils.Log.logJson(metadata)
  end

  if not opt.json_log then
    print('Building model...')
  end

  local model

  onmt.utils.Parallel.launch(nil, function(idx)

    _G.model = {}

    if checkpoint.models then
      _G.model.encoder = onmt.Models.loadEncoder(checkpoint.models.encoder, idx > 1)
      _G.model.decoder = onmt.Models.loadDecoder(checkpoint.models.decoder, idx > 1)
    else
      local verbose = idx == 1 and not opt.json_log
      _G.model.encoder = onmt.Models.buildEncoder(opt, dataset.dicts.src)
      _G.model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose)
    end

    for _, mod in pairs(_G.model) do
      onmt.utils.Cuda.convert(mod)
    end

    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)

  trainModel(model, trainData, validData, dataset, checkpoint.info)
end

main()
