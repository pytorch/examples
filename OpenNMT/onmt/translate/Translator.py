local checkpoint = nil
local models = {}
local dicts = {}
local opt = {}

local phraseTable

local function init(args)
  opt = args
  onmt.utils.Cuda.init(opt)

  print('Loading \'' .. opt.model .. '\'...')
  checkpoint = torch.load(opt.model)

  models.encoder = onmt.Models.loadEncoder(checkpoint.models.encoder)
  models.decoder = onmt.Models.loadDecoder(checkpoint.models.decoder)

  models.encoder:evaluate()
  models.decoder:evaluate()

  onmt.utils.Cuda.convert(models.encoder)
  onmt.utils.Cuda.convert(models.decoder)

  dicts = checkpoint.dicts

  if opt.phrase_table:len() > 0 then
    phraseTable = onmt.translate.PhraseTable.new(opt.phrase_table)
  end
end

local function buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local tgtData
  if goldBatch ~= nil then
    tgtData = {}
    tgtData.words = {}
    tgtData.features = {}
  end

  for b = 1, #srcBatch do
    table.insert(srcData.words, dicts.src.words:convertToIdx(srcBatch[b], onmt.Constants.UNK_WORD))

    if #dicts.src.features > 0 then
      table.insert(srcData.features,
                   onmt.utils.Features.generateSource(dicts.src.features, srcFeaturesBatch[b]))
    end

    if tgtData ~= nil then
      table.insert(tgtData.words,
                   dicts.tgt.words:convertToIdx(goldBatch[b],
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD))

      if #dicts.tgt.features > 0 then
        table.insert(tgtData.features,
                     onmt.utils.Features.generateTarget(dicts.tgt.features, goldFeaturesBatch[b]))
      end
    end
  end

  return onmt.data.Dataset.new(srcData, tgtData)
end

local function buildTargetTokens(pred, predFeats, src, attn)
  local tokens = dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  -- Always ignore last token to stay consistent, even it may not be EOS.
  table.remove(tokens)

  if opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == onmt.Constants.UNK_WORD then
        local _, maxIndex = attn[i]:max(1)
        local source = src[maxIndex[1]]

        if phraseTable and phraseTable:contains(source) then
          tokens[i] = phraseTable:lookup(source)
        else
          tokens[i] = source
        end
      end
    end
  end

  if predFeats ~= nil then
    tokens = onmt.utils.Features.annotate(tokens, predFeats, dicts.tgt.features)
  end

  return tokens
end

local function translateBatch(batch)
  models.encoder:maskPadding()
  models.decoder:maskPadding()

  local encStates, context = models.encoder:forward(batch)

  local goldScore
  if batch.targetInput ~= nil then
    if batch.size > 1 then
      models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
    end
    goldScore = models.decoder:computeScore(batch, encStates, context)
  end

  -- Expand tensors for each beam.
  context = context
    :contiguous()
    :view(1, batch.size, batch.sourceLength, checkpoint.options.rnn_size)
    :expand(opt.beam_size, batch.size, batch.sourceLength, checkpoint.options.rnn_size)
    :contiguous()
    :view(opt.beam_size * batch.size, batch.sourceLength, checkpoint.options.rnn_size)

  for j = 1, #encStates do
    encStates[j] = encStates[j]
      :view(1, batch.size, checkpoint.options.rnn_size)
      :expand(opt.beam_size, batch.size, checkpoint.options.rnn_size)
      :contiguous()
      :view(opt.beam_size * batch.size, checkpoint.options.rnn_size)
  end

  local remainingSents = batch.size

  -- As finished sentences are removed from the batch, this table maps the batches
  -- to their index within the remaining sentences.
  local batchIdx = {}

  local beam = {}

  for b = 1, batch.size do
    table.insert(beam, onmt.translate.Beam.new(opt.beam_size, #dicts.tgt.features))
    table.insert(batchIdx, b)
  end

  local i = 1

  local decOut
  local decStates = encStates

  while remainingSents > 0 and i < opt.max_sent_length do
    i = i + 1

    -- Prepare decoder input.
    local input = torch.IntTensor(opt.beam_size, remainingSents)
    local inputFeatures = {}
    local sourceSizes = torch.IntTensor(remainingSents)

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batchIdx[b]
        sourceSizes[idx] = batch.sourceSize[b]

        -- Get current state of the beam search.
        local wordState, featuresState = beam[b]:getCurrentState()
        input[{{}, idx}]:copy(wordState)

        for j = 1, #dicts.tgt.features do
          if inputFeatures[j] == nil then
            inputFeatures[j] = torch.IntTensor(opt.beam_size, remainingSents)
          end
          inputFeatures[j][{{}, idx}]:copy(featuresState[j])
        end
      end
    end

    input = input:view(opt.beam_size * remainingSents)
    for j = 1, #dicts.tgt.features do
      inputFeatures[j] = inputFeatures[j]:view(opt.beam_size * remainingSents)
    end

    local inputs
    if #inputFeatures == 0 then
      inputs = input
    else
      inputs = {}
      table.insert(inputs, input)
      onmt.utils.Table.append(inputs, inputFeatures)
    end

    if batch.size > 1 then
      models.decoder:maskPadding(sourceSizes, batch.sourceLength, opt.beam_size)
    end

    decOut, decStates = models.decoder:forwardOne(inputs, decStates, context, decOut)

    local out = models.decoder.generator:forward(decOut)

    for j = 1, #out do
      out[j] = out[j]:view(opt.beam_size, remainingSents, out[j]:size(2)):transpose(1, 2):contiguous()
    end
    local wordLk = out[1]

    local softmaxOut = models.decoder.softmaxAttn.output:view(opt.beam_size, remainingSents, -1)
    local newRemainingSents = remainingSents

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batchIdx[b]

        local featsLk = {}
        for j = 1, #dicts.tgt.features do
          table.insert(featsLk, out[j + 1][idx])
        end

        if beam[b]:advance(wordLk[idx], featsLk, softmaxOut[{{}, idx}]) then
          newRemainingSents = newRemainingSents - 1
          batchIdx[b] = 0
        end

        for j = 1, #decStates do
          local view = decStates[j]
            :view(opt.beam_size, remainingSents, checkpoint.options.rnn_size)
          view[{{}, idx}] = view[{{}, idx}]:index(1, beam[b]:getCurrentOrigin())
        end
      end
    end

    if newRemainingSents > 0 and newRemainingSents ~= remainingSents then
      -- Update sentence indices within the batch and mark sentences to keep.
      local toKeep = {}
      local newIdx = 1
      for b = 1, #batchIdx do
        local idx = batchIdx[b]
        if idx > 0 then
          table.insert(toKeep, idx)
          batchIdx[b] = newIdx
          newIdx = newIdx + 1
        end
      end

      toKeep = torch.LongTensor(toKeep)

      -- Update rnn states and context.
      for j = 1, #decStates do
        decStates[j] = decStates[j]
          :view(opt.beam_size, remainingSents, checkpoint.options.rnn_size)
          :index(2, toKeep)
          :view(opt.beam_size*newRemainingSents, checkpoint.options.rnn_size)
      end

      decOut = decOut
        :view(opt.beam_size, remainingSents, checkpoint.options.rnn_size)
        :index(2, toKeep)
        :view(opt.beam_size*newRemainingSents, checkpoint.options.rnn_size)

      context = context
        :view(opt.beam_size, remainingSents, batch.sourceLength, checkpoint.options.rnn_size)
        :index(2, toKeep)
        :view(opt.beam_size*newRemainingSents, batch.sourceLength, checkpoint.options.rnn_size)

      -- The `index()` method allocates a new storage so clean the previous ones to
      -- keep a stable memory usage.
      collectgarbage()
    end

    remainingSents = newRemainingSents
  end

  local allHyp = {}
  local allFeats = {}
  local allAttn = {}
  local allScores = {}

  for b = 1, batch.size do
    local scores, ks = beam[b]:sortBest()

    local hypBatch = {}
    local featsBatch = {}
    local attnBatch = {}
    local scoresBatch = {}

    for n = 1, opt.n_best do
      local hyp, feats, attn = beam[b]:getHyp(ks[n])

      -- remove unnecessary values from the attention vectors
      for j = 1, #attn do
        local size = batch.sourceSize[b]
        attn[j] = attn[j]:narrow(1, batch.sourceLength - size + 1, size)
      end

      table.insert(hypBatch, hyp)
      if #feats > 0 then
        table.insert(featsBatch, feats)
      end
      table.insert(attnBatch, attn)
      table.insert(scoresBatch, scores[n])
    end

    table.insert(allHyp, hypBatch)
    table.insert(allFeats, featsBatch)
    table.insert(allAttn, attnBatch)
    table.insert(allScores, scoresBatch)
  end

  return allHyp, allFeats, allScores, allAttn, goldScore
end

local function translate(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local data = buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local batch = data:getBatch()

  local pred, predFeats, predScore, attn, goldScore = translateBatch(batch)

  local predBatch = {}
  local infoBatch = {}

  for b = 1, batch.size do
    table.insert(predBatch, buildTargetTokens(pred[b][1], predFeats[b][1], srcBatch[b], attn[b][1]))

    local info = {}
    info.score = predScore[b][1]
    info.nBest = {}

    if goldScore ~= nil then
      info.goldScore = goldScore[b]
    end

    if opt.n_best > 1 then
      for n = 1, opt.n_best do
        info.nBest[n] = {}
        info.nBest[n].tokens = buildTargetTokens(pred[b][n], predFeats[b][n], srcBatch[b], attn[b][n])
        info.nBest[n].score = predScore[b][n]
      end
    end

    table.insert(infoBatch, info)
  end

  return predBatch, infoBatch
end

return {
  init = init,
  translate = translate
}
