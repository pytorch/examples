--[[ nn unit. Maps from word ids to embeddings. Slim wrapper around
nn.LookupTable to allow fixed and pretrained embeddings.
--]]
local WordEmbedding, parent = torch.class('onmt.WordEmbedding', 'nn.Container')

--[[
Parameters:

  * `vocabSize` - size of the vocabulary
  * `vecSize` - size of the embedding
  * `preTrainined` - path to a pretrained vector file
  * `fix` - keep the weights of the embeddings fixed.
--]]
function WordEmbedding:__init(vocabSize, vecSize, preTrained, fix)
  parent.__init(self)
  self.vocabSize = vocabSize
  self.net = nn.LookupTable(vocabSize, vecSize, onmt.Constants.PAD)
  self:add(self.net)

  -- If embeddings are given. Initialize them.
  if preTrained and preTrained:len() > 0 then
    local vecs = torch.load(preTrained)
    self.net.weight:copy(vecs)

    self.fix = fix
    if self.fix then
      self.net.gradWeight = nil
    end
  end
end

function WordEmbedding:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function WordEmbedding:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function WordEmbedding:accGradParameters(input, gradOutput, scale)
  if not self.fix then
    self.net:accGradParameters(input, gradOutput, scale)
  end
end

function WordEmbedding:parameters()
  if not self.fix then
    return parent.parameters(self)
  end
end
