--[[Helper function convert a `flatIndex` to a row-column tuple

Parameters:

  * `v` - matrix.
  * `flatIndex` - index

Returns: row/column.
--]]
local function flatToRc(v, flatIndex)
  local row = math.floor((flatIndex - 1) / v:size(2)) + 1
  return row, (flatIndex - 1) % v:size(2) + 1
end

--[[ Class for managing the internals of the beam search process.


    hyp1---hyp1---hyp1 -hyp1
        \             /
    hyp2 \-hyp2 /-hyp2--hyp2
               /      \
    hyp3---hyp3---hyp3 -hyp3
    ========================

Takes care of beams, back pointers, and scores.
]]
local Beam = torch.class('Beam')

--[[Constructor

Parameters:

  * `size` : The beam `K`.
  * `numFeatures` : Number of features, (optional)
--]]
function Beam:__init(size, numFeatures)

  self.size = size
  self.numFeatures = numFeatures
  self.done = false

  -- The score for each translation on the beam.
  self.scores = torch.FloatTensor(size):zero()

  -- The backpointers at each time-step.
  self.prevKs = { torch.LongTensor(size):fill(1) }

  -- The outputs at each time-step.
  self.nextYs = { torch.LongTensor(size):fill(onmt.Constants.PAD) }
  self.nextYs[1][1] = onmt.Constants.BOS

  -- The features output at each time-step
  self.nextFeatures = { {} }
  for j = 1, numFeatures do
    self.nextFeatures[1][j] = torch.LongTensor(size):fill(onmt.Constants.PAD)

    -- EOS is used as a placeholder to shift the features target sequence.
    self.nextFeatures[1][j][1] = onmt.Constants.EOS
  end

  -- The attentions (matrix) for each time.
  self.attn = {}
end

--[[ Get the outputs for the current timestep.]]
function Beam:getCurrentState()
  return self.nextYs[#self.nextYs], self.nextFeatures[#self.nextFeatures]
end

--[[ Get the backpointers for the current timestep.]]
function Beam:getCurrentOrigin()
  return self.prevKs[#self.prevKs]
end

--[[ Given prob over words for every last beam `wordLk` and attention
 `attnOut`. Compute and update the beam search.

Parameters:

  * `wordLk`- probs of advancing from the last step (K x words)
  * `featsLk`- probs of features at the last step (K x numfeatures x featsize)
  * `attnOut`- attention at the last step

Returns: true if beam search is complete.
--]]
function Beam:advance(wordLk, featsLk, attnOut)

  -- The flattened scores.
  local flatWordLk

  if #self.prevKs > 1 then
    -- Sum the previous scores.
    for k = 1, self.size do
      wordLk[k]:add(self.scores[k])
    end
    flatWordLk = wordLk:view(-1)
  else
    flatWordLk = wordLk[1]:view(-1)
  end


  -- Find the top-k elements in flatWordLk and backpointers.
  local prevK = torch.LongTensor(self.size)
  local nextY = torch.LongTensor(self.size)
  local nextFeat = {}
  local attn = {}

  for j = 1, #featsLk do
    nextFeat[j] = torch.LongTensor(self.size)
  end

  local bestScores, bestScoresId = flatWordLk:topk(self.size, 1, true, true)

  for k = 1, self.size do
    self.scores[k] = bestScores[k]

    local fromBeam, bestScoreId = flatToRc(wordLk, bestScoresId[k])

    prevK[k] = fromBeam
    nextY[k] = bestScoreId
    table.insert(attn, attnOut[fromBeam]:clone())

    -- For features, just store predictions for each beam.
    for j = 1, #featsLk do
      local _, best = featsLk[j]:max(2)
      nextFeat[j]:copy(best)
    end
  end

  -- End condition is when top-of-beam is EOS.
  if nextY[1] == onmt.Constants.EOS then
    self.done = true
  end

  table.insert(self.prevKs, prevK)
  table.insert(self.nextYs, nextY)
  table.insert(self.nextFeatures, nextFeat)
  table.insert(self.attn, attn)

  return self.done
end

function Beam:sortBest()
  return torch.sort(self.scores, 1, true)
end

--[[ Get the score of the best in the beam. ]]
function Beam:getBest()
  local scores, ids = self:sortBest()
  return scores[1], ids[1]
end

--[[ Walk back to construct the full hypothesis.

Parameters:

  * `k` - the position in the beam to construct.

Returns:

  1. The hypothesis
  2. The attention at each time step.
--]]
function Beam:getHyp(k)
  local hyp = {}
  local feats = {}
  local attn = {}

  for _ = 1, #self.prevKs - 1 do
    table.insert(hyp, {})
    table.insert(attn, {})

    if self.numFeatures > 0 then
      table.insert(feats, {})
    end
  end

  for j = #self.prevKs, 2, -1 do
    hyp[j - 1] = self.nextYs[j][k]
    for i = 1, self.numFeatures do
      feats[j - 1][i] = self.nextFeatures[j][i][k]
    end
    attn[j - 1] = self.attn[j - 1][k]
    k = self.prevKs[j][k]
  end

  return hyp, feats, attn
end

return Beam
