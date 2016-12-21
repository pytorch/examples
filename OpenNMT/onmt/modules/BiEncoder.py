local function reverseInput(batch)
  batch.sourceInput, batch.sourceInputRev = batch.sourceInputRev, batch.sourceInput
  batch.sourceInputFeatures, batch.sourceInputRevFeatures = batch.sourceInputRevFeatures, batch.sourceInputFeatures
  batch.sourceInputPadLeft, batch.sourceInputRevPadLeft = batch.sourceInputRevPadLeft, batch.sourceInputPadLeft
end

--[[ BiEncoder is a bidirectional Sequencer used for the source language.


 `netFwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

 `netBwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local BiEncoder, parent = torch.class('onmt.BiEncoder', 'nn.Container')

--[[ Create a bi-encoder.

Parameters:

  * `input` - input neural network.
  * `rnn` - recurrent template module.
  * `merge` - fwd/bwd merge operation {"concat", "sum"}
]]
function BiEncoder:__init(input, rnn, merge)
  parent.__init(self)

  self.fwd = onmt.Encoder.new(input, rnn)
  self.bwd = onmt.Encoder.new(input:clone('weight', 'bias', 'gradWeight', 'gradBias'), rnn:clone())

  self.args = {}
  self.args.merge = merge

  self.args.rnnSize = rnn.outputSize
  self.args.numEffectiveLayers = rnn.numEffectiveLayers

  if self.args.merge == 'concat' then
    self.args.hiddenSize = self.args.rnnSize * 2
  else
    self.args.hiddenSize = self.args.rnnSize
  end

  self:add(self.fwd)
  self:add(self.bwd)

  self:resetPreallocation()
end

--[[ Return a new BiEncoder using the serialized data `pretrained`. ]]
function BiEncoder.load(pretrained)
  local self = torch.factory('onmt.BiEncoder')()

  parent.__init(self)

  self.fwd = onmt.Encoder.load(pretrained.modules[1])
  self.bwd = onmt.Encoder.load(pretrained.modules[2])
  self.args = pretrained.args

  self:add(self.fwd)
  self:add(self.bwd)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function BiEncoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function BiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
end

function BiEncoder:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function BiEncoder:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, self.args.hiddenSize })

  local fwdStates, fwdContext = self.fwd:forward(batch)
  reverseInput(batch)
  local bwdStates, bwdContext = self.bwd:forward(batch)
  reverseInput(batch)

  if self.args.merge == 'concat' then
    for i = 1, #fwdStates do
      states[i]:narrow(2, 1, self.args.rnnSize):copy(fwdStates[i])
      states[i]:narrow(2, self.args.rnnSize + 1, self.args.rnnSize):copy(bwdStates[i])
    end
    for t = 1, batch.sourceLength do
      context[{{}, t}]:narrow(2, 1, self.args.rnnSize)
        :copy(fwdContext[{{}, t}])
      context[{{}, t}]:narrow(2, self.args.rnnSize + 1, self.args.rnnSize)
        :copy(bwdContext[{{}, batch.sourceLength - t + 1}])
    end
  elseif self.args.merge == 'sum' then
    for i = 1, #states do
      states[i]:copy(fwdStates[i])
      states[i]:add(bwdStates[i])
    end
    for t = 1, batch.sourceLength do
      context[{{}, t}]:copy(fwdContext[{{}, t}])
      context[{{}, t}]:add(bwdContext[{{}, batch.sourceLength - t + 1}])
    end
  end

  return states, context
end

function BiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  gradStatesOutput = gradStatesOutput
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize*2 })

  local gradContextOutputFwd
  local gradContextOutputBwd

  local gradStatesOutputFwd = {}
  local gradStatesOutputBwd = {}

  if self.args.merge == 'concat' then
    local gradContextOutputSplit = gradContextOutput:chunk(2, 3)
    gradContextOutputFwd = gradContextOutputSplit[1]
    gradContextOutputBwd = gradContextOutputSplit[2]

    for i = 1, #gradStatesOutput do
      local statesSplit = gradStatesOutput[i]:chunk(2, 2)
      table.insert(gradStatesOutputFwd, statesSplit[1])
      table.insert(gradStatesOutputBwd, statesSplit[2])
    end
  elseif self.args.merge == 'sum' then
    gradContextOutputFwd = gradContextOutput
    gradContextOutputBwd = gradContextOutput

    gradStatesOutputFwd = gradStatesOutput
    gradStatesOutputBwd = gradStatesOutput
  end

  local gradInputFwd = self.fwd:backward(batch, gradStatesOutputFwd, gradContextOutputFwd)

  -- reverse gradients of the backward context
  local gradContextBwd = onmt.utils.Tensor.reuseTensor(self.gradContextBwdProto,
                                                       { batch.size, batch.sourceLength, self.args.rnnSize })

  for t = 1, batch.sourceLength do
    gradContextBwd[{{}, t}]:copy(gradContextOutputBwd[{{}, batch.sourceLength - t + 1}])
  end

  local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd, gradContextBwd)

  for t = 1, batch.sourceLength do
    local revIndex = batch.sourceLength - t + 1
    if torch.isTensor(gradInputFwd[t]) then
      gradInputFwd[t]:add(gradInputBwd[revIndex])
    else
      for i = 1, #gradInputFwd[t] do
        gradInputFwd[t][i]:add(gradInputBwd[revIndex][i])
      end
    end
  end

  return gradInputFwd
end
