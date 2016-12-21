--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class("EpochState")

--[[ Initialize for epoch `epoch` and training `status` (current loss)]]
function EpochState:__init(epoch, numIterations, learningRate, lastValidPpl, status)
  self.epoch = epoch
  self.numIterations = numIterations
  self.learningRate = learningRate
  self.lastValidPpl = lastValidPpl

  if status ~= nil then
    self.status = status
  else
    self.status = {}
    self.status.trainNonzeros = 0
    self.status.trainLoss = 0
  end

  self.timer = torch.Timer()
  self.numWordsSource = 0
  self.numWordsTarget = 0

  self.minFreeMemory = 100000000000
end

--[[ Update training status. Takes `batch` (described in data.lua) and last losses.]]
function EpochState:update(batches, losses)
  for i = 1,#batches do
    self.numWordsSource = self.numWordsSource + batches[i].size * batches[i].sourceLength
    self.numWordsTarget = self.numWordsTarget + batches[i].size * batches[i].targetLength
    self.status.trainLoss = self.status.trainLoss + losses[i]
    self.status.trainNonzeros = self.status.trainNonzeros + batches[i].targetNonZeros
  end
end

--[[ Log to status stdout. ]]
function EpochState:log(batchIndex, json)
  if json then
    local freeMemory = onmt.utils.Cuda.freeMemory()
    if freeMemory < self.minFreeMemory then
      self.minFreeMemory = freeMemory
    end

    local obj = {
      time = os.time(),
      epoch = self.epoch,
      iteration = batchIndex,
      totalIterations = self.numIterations,
      learningRate = self.learningRate,
      trainingPerplexity = self:getTrainPpl(),
      freeMemory = freeMemory,
      lastValidationPerplexity = self.lastValidPpl,
      processedTokens = {
        source = self.numWordsSource,
        target = self.numWordsTarget
      }
    }

    onmt.utils.Log.logJson(obj)
  else
    local timeTaken = self:getTime()

    local stats = ''
    stats = stats .. string.format('Epoch %d ; ', self.epoch)
    stats = stats .. string.format('Iteration %d/%d ; ', batchIndex, self.numIterations)
    stats = stats .. string.format('Learning rate %.4f ; ', self.learningRate)
    stats = stats .. string.format('Source tokens/s %d ; ', self.numWordsSource / timeTaken)
    stats = stats .. string.format('Perplexity %.2f', self:getTrainPpl())
    print(stats)
  end
end

function EpochState:getTrainPpl()
  return math.exp(self.status.trainLoss / self.status.trainNonzeros)
end

function EpochState:getTime()
  return self.timer:time().real
end

function EpochState:getStatus()
  return self.status
end

function EpochState:getMinFreememory()
  return self.minFreeMemory
end

return EpochState
