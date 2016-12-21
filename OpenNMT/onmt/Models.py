local function buildEncoder(opt, dicts)
  local inputNetwork = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                              opt.word_vec_size,
                                              opt.pre_word_vecs_enc,
                                              opt.fix_word_vecs_enc)

  local inputSize = opt.word_vec_size

  -- Sequences with features.
  if #dicts.features > 0 then
    local srcFeatEmbedding = onmt.FeaturesEmbedding.new(dicts.features,
                                                        opt.feat_vec_exponent,
                                                        opt.feat_vec_size,
                                                        opt.feat_merge)

    inputNetwork = nn.Sequential()
      :add(nn.ParallelTable()
             :add(inputNetwork)
             :add(srcFeatEmbedding))
      :add(nn.JoinTable(2))

    inputSize = inputSize + srcFeatEmbedding.outputSize
  end

  if opt.brnn then
    -- Compute rnn hidden size depending on hidden states merge action.
    local rnnSize = opt.rnn_size
    if opt.brnn_merge == 'concat' then
      if opt.rnn_size % 2 ~= 0 then
        error('in concat mode, rnn_size must be divisible by 2')
      end
      rnnSize = rnnSize / 2
    elseif opt.brnn_merge == 'sum' then
      rnnSize = rnnSize
    else
      error('invalid merge action ' .. opt.brnn_merge)
    end

    local rnn = onmt.LSTM.new(opt.layers, inputSize, rnnSize, opt.dropout, opt.residual)

    return onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
  else
    local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

    return onmt.Encoder.new(inputNetwork, rnn)
  end
end

local function buildDecoder(opt, dicts, verbose)
  local inputNetwork = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                              opt.word_vec_size,
                                              opt.pre_word_vecs_dec,
                                              opt.fix_word_vecs_dec)

  local inputSize = opt.word_vec_size

  local generator

  -- Sequences with features.
  if #dicts.features > 0 then
    local tgtFeatEmbedding = onmt.FeaturesEmbedding.new(dicts.features,
                                                        opt.feat_vec_exponent,
                                                        opt.feat_vec_size,
                                                        opt.feat_merge)

    inputNetwork = nn.Sequential()
      :add(nn.ParallelTable()
             :add(inputNetwork)
             :add(tgtFeatEmbedding))
      :add(nn.JoinTable(2))

    inputSize = inputSize + tgtFeatEmbedding.outputSize

    generator = onmt.FeaturesGenerator.new(opt.rnn_size, dicts.words:size(), dicts.features)
  else
    generator = onmt.Generator.new(opt.rnn_size, dicts.words:size())
  end

  if opt.input_feed == 1 then
    if verbose then
      print(" * using input feeding")
    end
    inputSize = inputSize + opt.rnn_size
  end

  local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end

--[[ This is useful when training from a model in parallel mode: each thread must own its model. ]]
local function clonePretrained(model)
  local clone = {}

  for k, v in pairs(model) do
    if k == 'modules' then
      clone.modules = {}
      for i = 1, #v do
        table.insert(clone.modules, onmt.utils.Tensor.deepClone(v[i]))
      end
    else
      clone[k] = v
    end
  end

  return clone
end

local function loadEncoder(pretrained, clone)
  local brnn = #pretrained.modules == 2

  if clone then
    pretrained = clonePretrained(pretrained)
  end

  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

local function loadDecoder(pretrained, clone)
  if clone then
    pretrained = clonePretrained(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end

return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
