require('nngraph')

--[[ A batched-softmax wrapper to mask the probabilities of padding.

  For instance there may be a batch of instances where A is padding.

    AXXXAA
    AXXAAA
    AXXXXX

  MaskedSoftmax ensures that no probability is given to the A's.

  For this example, `beamSize` is 3, `sourceLength` is {3, 2, 5}.
--]]
local MaskedSoftmax, parent = torch.class('onmt.MaskedSoftmax', 'nn.Container')


--[[ A nn-style module that applies a softmax on input that gives no weight to the left padding.

Parameters:

  * `sourceSizes` -  the true lengths (with left padding).
  * `sourceLength` - the max length in the batch `beamSize`.
  * `beamSize` - the batch size.
--]]
function MaskedSoftmax:__init(sourceSizes, sourceLength, beamSize)
  parent.__init(self)
  --TODO: better names for these variables. Beam size =? batchSize?
  self.net = self:_buildModel(sourceSizes, sourceLength, beamSize)
  self:add(self.net)
end

function MaskedSoftmax:_buildModel(sourceSizes, sourceLength, beamSize)

  local numSents = sourceSizes:size(1)
  local input = nn.Identity()()
  local softmax = nn.SoftMax()(input) -- beamSize*numSents x State.sourceLength

  -- Now we are masking the part of the output we don't need
  local tab
  if beamSize ~= nil then
    tab = nn.SplitTable(2)(nn.View(beamSize, numSents, sourceLength)(softmax))
    -- numSents x { beamSize x State.sourceLength }
  else
    tab = nn.SplitTable(1)(softmax) -- numSents x { State.sourceLength }
  end

  local par = nn.ParallelTable()

  for b = 1, numSents do
    local padLength = sourceLength - sourceSizes[b]
    local dim = 2
    if beamSize == nil then
      dim = 1
    end

    local seq = nn.Sequential()
    seq:add(nn.Narrow(dim, padLength + 1, sourceSizes[b]))
    seq:add(nn.Padding(1, -padLength, 1, 0))
    par:add(seq)
  end

  local outTab = par(tab) -- numSents x { beamSize x State.sourceLength }
  local output = nn.JoinTable(1)(outTab) -- numSents*beamSize x State.sourceLength
  if beamSize ~= nil then
    output = nn.View(numSents, beamSize, sourceLength)(output)
    output = nn.Transpose({1,2})(output) -- beamSize x numSents x State.sourceLength
    output = nn.View(beamSize*numSents, sourceLength)(output)
  else
    output = nn.View(numSents, sourceLength)(output)
  end

  -- Make sure the vector sums to 1 (softmax output)
  output = nn.Normalize(1)(output)

  return nn.gModule({input}, {output})
end

function MaskedSoftmax:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function MaskedSoftmax:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function MaskedSoftmax:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
