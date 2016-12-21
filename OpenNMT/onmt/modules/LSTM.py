require('nngraph')

--[[
Implementation of a single stacked-LSTM step as
an nn unit.

      h^L_{t-1} --- h^L_t
      c^L_{t-1} --- c^L_t
                 |


                 .
                 |
             [dropout]
                 |
      h^1_{t-1} --- h^1_t
      c^1_{t-1} --- c^1_t
                 |
                 |
                x_t

Computes $$(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})$$.

--]]
local LSTM, parent = torch.class('onmt.LSTM', 'nn.Container')

--[[
Parameters:

  * `layers` - Number of LSTM layers, L.
  * `inputSize` - Size of input layer
  * `hiddenSize` - Size of the hidden layers.
  * `dropout` - Dropout rate to use.
  * `residual` - Residual connections between layers.
--]]
function LSTM:__init(layers, inputSize, hiddenSize, dropout, residual)
  parent.__init(self)

  dropout = dropout or 0

  self.dropout = dropout
  self.numEffectiveLayers = 2 * layers
  self.outputSize = hiddenSize

  self.net = self:_buildModel(layers, inputSize, hiddenSize, dropout, residual)
  self:add(self.net)
end

--[[ Stack the LSTM units. ]]
function LSTM:_buildModel(layers, inputSize, hiddenSize, dropout, residual)
  local inputs = {}
  local outputs = {}

  for _ = 1, layers do
    table.insert(inputs, nn.Identity()()) -- c0: batchSize x hiddenSize
    table.insert(inputs, nn.Identity()()) -- h0: batchSize x hiddenSize
  end

  table.insert(inputs, nn.Identity()()) -- x: batchSize x inputSize
  local x = inputs[#inputs]

  local prevInput
  local nextC
  local nextH

  for L = 1, layers do
    local input
    local inputDim

    if L == 1 then
      -- First layer input is x.
      input = x
      inputDim = inputSize
    else
      inputDim = hiddenSize
      input = nextH
      if residual and (L > 2 or inputSize == hiddenSize) then
        input = nn.CAddTable()({input, prevInput})
      end
      if dropout > 0 then
        input = nn.Dropout(dropout)(input)
      end
    end

    local prevC = inputs[L*2 - 1]
    local prevH = inputs[L*2]

    nextC, nextH = self:_buildLayer(inputDim, hiddenSize)({prevC, prevH, input}):split(2)
    prevInput = input

    table.insert(outputs, nextC)
    table.insert(outputs, nextH)
  end

  return nn.gModule(inputs, outputs)
end

--[[ Build a single LSTM unit layer. ]]
function LSTM:_buildLayer(inputSize, hiddenSize)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prevC = inputs[1]
  local prevH = inputs[2]
  local x = inputs[3]

  -- Evaluate the input sums at once for efficiency.
  local i2h = nn.Linear(inputSize, 4 * hiddenSize)(x)
  local h2h = nn.Linear(hiddenSize, 4 * hiddenSize, false)(prevH)
  local allInputSums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, hiddenSize)(allInputSums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

  -- Decode the gates.
  local inGate = nn.Sigmoid()(n1)
  local forgetGate = nn.Sigmoid()(n2)
  local outGate = nn.Sigmoid()(n3)

  -- Decode the write inputs.
  local inTransform = nn.Tanh()(n4)

  -- Perform the LSTM update.
  local nextC = nn.CAddTable()({
    nn.CMulTable()({forgetGate, prevC}),
    nn.CMulTable()({inGate, inTransform})
  })

  -- Gated cells form the output.
  local nextH = nn.CMulTable()({outGate, nn.Tanh()(nextC)})

  return nn.gModule(inputs, {nextC, nextH})
end

function LSTM:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function LSTM:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function LSTM:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
