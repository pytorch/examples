#[[ Unit to decode a sequence of output tokens.

          .      .      .             .
          |      |      |             |
        h_1 => h_2 => h_3 => ... => h_n
          |      |      |             |
          .      .      .             .
          |      |      |             |
        h_1 => h_2 => h_3 => ... => h_n
          |      |      |             |
          |      |      |             |
        x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

#]]
Decoder, parent = torch.class('onmt.Decoder', 'onmt.Sequencer')


#[[ Construct a decoder layer.

Parameters.

    * `inputNetwork` - input nn module.
    * `rnn` - recurrent module, such as [onmt.LSTM](onmt+modules+LSTM).
    * `generator` - optional, an output [onmt.Generator](onmt+modules+Generator).
    * `inputFeed` - bool, enable input feeding.
#]]
def Decoder.__init(inputNetwork, rnn, generator, inputFeed):
    self.rnn = rnn
    self.inputNet = inputNetwork

    self.args = {}
    self.args.rnnSize = self.rnn.outputSize
    self.args.numEffectiveLayers = self.rnn.numEffectiveLayers

    self.args.inputIndex = {}
    self.args.outputIndex = {}

    # Input feeding means the decoder takes an extra
    # vector each time representing the attention at the
    # previous step.
    self.args.inputFeed = inputFeed

    parent.__init(self, self._buildModel())

    # The generator use the output of the decoder sequencer to generate the
    # likelihoods over the target vocabulary.
    self.generator = generator
    self.add(self.generator)

    self.resetPreallocation()


#" Return a new Decoder using the serialized data `pretrained`. "
def Decoder.load(pretrained):
    self = torch.factory('onmt.Decoder')()

    self.args = pretrained.args

    parent.__init(self, pretrained.modules[1])
    self.generator = pretrained.modules[2]
    self.add(self.generator)

    self.resetPreallocation()

    return self


#" Return data to serialize. "
def Decoder.serialize():
    return {
        modules = self.modules,
        args = self.args
    }


def Decoder.resetPreallocation():
    if self.args.inputFeed:
        self.inputFeedProto = torch.Tensor()
    

    # Prototype for preallocated hidden and cell states.
    self.stateProto = torch.Tensor()

    # Prototype for preallocated output gradients.
    self.gradOutputProto = torch.Tensor()

    # Prototype for preallocated context gradient.
    self.gradContextProto = torch.Tensor()


#[[ Build a default one time-step of the decoder

Returns. An nn-graph mapping

    $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
    (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

    Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
    ${x_t}$ is a sparse word to lookup,
    ${con/H}$ is the context/source hidden states for attention,
    ${if}$ is the input feeding, and
    ${a}$ is the context vector computed at this timestep.
#]]
def Decoder._buildModel():
    inputs = {}
    states = {}

    # Inputs are previous layers first.
    for _ = 1, self.args.numEffectiveLayers:
        h0 = nn.Identity()() # batchSize x rnnSize
        table.insert(inputs, h0)
        table.insert(states, h0)
    

    x = nn.Identity()() # batchSize
    table.insert(inputs, x)
    self.args.inputIndex.x = len(inputs)

    context = nn.Identity()() # batchSize x sourceLength x rnnSize
    table.insert(inputs, context)
    self.args.inputIndex.context = len(inputs)

    inputFeed
    if self.args.inputFeed:
        inputFeed = nn.Identity()() # batchSize x rnnSize
        table.insert(inputs, inputFeed)
        self.args.inputIndex.inputFeed = len(inputs)
    

    # Compute the input network.
    input = self.inputNet(x)

    # If set, concatenate previous decoder output.
    if self.args.inputFeed:
        input = nn.JoinTable(2)({input, inputFeed})
    
    table.insert(states, input)

    # Forward states and input into the RNN.
    outputs = self.rnn(states)

    # The output of a subgraph is a node. split it to access the last RNN output.
    outputs = { outputs.split(self.args.numEffectiveLayers) }

    # Compute the attention here using h^L as query.
    attnLayer = onmt.GlobalAttention(self.args.rnnSize)
    attnLayer.name = 'decoderAttn'
    attnOutput = attnLayer({outputs[len(outputs]), context})
    if self.rnn.dropout > 0:
        attnOutput = nn.Dropout(self.rnn.dropout)(attnOutput)
    
    table.insert(outputs, attnOutput)
    return nn.gModule(inputs, outputs)


#[[ Mask padding means that the attention-layer is constrained to
    give zero-weight to padding. This is done by storing a reference
    to the softmax attention-layer.

    Parameters.

    * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
#]]
def Decoder.maskPadding(sourceSizes, sourceLength, beamSize):
    if not self.decoderAttn:
        self.network.apply(def (layer):
            if layer.name == 'decoderAttn':
                self.decoderAttn = layer
            
        )
    

    self.decoderAttn.replace(function(module)
        if module.name == 'softmaxAttn':
            mod
            if sourceSizes != None:
                mod = onmt.MaskedSoftmax(sourceSizes, sourceLength, beamSize)
            else:
                mod = nn.SoftMax()
            

            mod.name = 'softmaxAttn'
            mod.type(module._type)
            self.softmaxAttn = mod
            return mod
        else:
            return module
        
    )


#[[ Run one step of the decoder.

Parameters.

    * `input` - input to be passed to inputNetwork.
    * `prevStates` - stack of hidden states (batch x layers*model x rnnSize)
    * `context` - encoder output (batch x n x rnnSize)
    * `prevOut` - previous distribution (batch x len(words))
    * `t` - current timestep

Returns.

  1. `out` - Top-layer hidden state.
  2. `states` - All states.
#]]
def Decoder.forwardOne(input, prevStates, context, prevOut, t):
    inputs = {}

    # Create RNN input (see sequencer.lua `buildNetwork('dec')`).
    onmt.utils.Table.app(inputs, prevStates)
    table.insert(inputs, input)
    table.insert(inputs, context)
    inputSize
    if torch.type(input) == 'table':
        inputSize = input[1].size(1)
    else:
        inputSize = input.size(1)
    

    if self.args.inputFeed:
        if prevOut == None:
            table.insert(inputs, onmt.utils.Tensor.reuseTensor(self.inputFeedProto,
                                                                                                                  { inputSize, self.args.rnnSize }))
        else:
            table.insert(inputs, prevOut)
        
    

    # Remember inputs for the backward pass.
    if self.train:
        self.inputs[t] = inputs
    

    outputs = self.net(t):forward(inputs)
    out = outputs[len(outputs])
    states = {}
    for i = 1, len(outputs) - 1:
        table.insert(states, outputs[i])
    

    return out, states


#[[Compute all forward steps.

    Parameters.

    * `batch` - `Batch` object
    * `encoderStates` -
    * `context` -
    * `func` - Calls `func(out, t)` each timestep.
#]]

def Decoder.forwardAndApply(batch, encoderStates, context, func):
    # TODO. Make this a private method.

    if self.statesProto == None:
        self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                                                                                  self.stateProto,
                                                                                                                  { batch.size, self.args.rnnSize })
    

    states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)

    prevOut

    for t = 1, batch.targetLength:
        prevOut, states = self.forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
        func(prevOut, t)
    


#[[Compute all forward steps.

    Parameters.

    * `batch` - a `Batch` object.
    * `encoderStates` - a batch of initial decoder states (optional) [0]
    * `context` - the context to apply attention to.

    Returns. Table of top hidden state for each timestep.
#]]
def Decoder.forward(batch, encoderStates, context):
    encoderStates = encoderStates
        or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                                                  onmt.utils.Cuda.convert(torch.Tensor()),
                                                                                  { batch.size, self.args.rnnSize })
    if self.train:
        self.inputs = {}
    

    outputs = {}

    self.forwardAndApply(batch, encoderStates, context, def (out):
        table.insert(outputs, out)
    )

    return outputs


#[[ Compute the backward update.

Parameters.

    * `batch` - a `Batch` object
    * `outputs` - expected outputs
    * `criterion` - a single target criterion object

    Note. This code runs both the standard backward and criterion forward/backward.
    It returns both the gradInputs and the loss.
    # ]]
def Decoder.backward(batch, outputs, criterion):
    if self.gradOutputsProto == None:
        self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers + 1,
                                                                                                                            self.gradOutputProto,
                                                                                                                            { batch.size, self.args.rnnSize })
    

    gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                                                                                          { batch.size, self.args.rnnSize })
    gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                                                                                  { batch.size, batch.sourceLength, self.args.rnnSize })

    loss = 0

    for t = batch.targetLength, 1, -1:
        # Compute decoder output gradients.
        # Note. This would typically be in the forward pass.
        pred = self.generator.forward(outputs[t])
        output = batch.getTargetOutput(t)

        loss = loss + criterion.forward(pred, output)

        # Compute the criterion gradient.
        genGradOut = criterion.backward(pred, output)
        for j = 1, len(genGradOut):
            genGradOut[j].div(batch.totalSize)
        

        # Compute the final layer gradient.
        decGradOut = self.generator.backward(outputs[t], genGradOut)
        gradStatesInput[len(gradStatesInput].add(decGradOut))

        # Compute the standarad backward.
        gradInput = self.net(t):backward(self.inputs[t], gradStatesInput)

        # Accumulate encoder output gradients.
        gradContextInput.add(gradInput[self.args.inputIndex.context])
        gradStatesInput[len(gradStatesInput].zero())

        # Accumulate previous output gradients with input feeding gradients.
        if self.args.inputFeed and t > 1:
            gradStatesInput[len(gradStatesInput].add(gradInput[self.args.inputIndex.inputFeed]))
        

        # Prepare next decoder output gradients.
        for i = 1, len(self.statesProto):
            gradStatesInput[i].copy(gradInput[i])
        
    

    return gradStatesInput, gradContextInput, loss


#[[ Compute the loss on a batch.

Parameters.

    * `batch` - a `Batch` to score.
    * `encoderStates` - initialization of decoder.
    * `context` - the attention context.
    * `criterion` - a pointwise criterion.

#]]
def Decoder.computeLoss(batch, encoderStates, context, criterion):
    encoderStates = encoderStates
        or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                                                  onmt.utils.Cuda.convert(torch.Tensor()),
                                                                                  { batch.size, self.args.rnnSize })

    loss = 0
    self.forwardAndApply(batch, encoderStates, context, def (out, t):
        pred = self.generator.forward(out)
        output = batch.getTargetOutput(t)
        loss = loss + criterion.forward(pred, output)
    )

    return loss



#[[ Compute the score of a batch.

Parameters.

    * `batch` - a `Batch` to score.
    * `encoderStates` - initialization of decoder.
    * `context` - the attention context.

#]]
def Decoder.computeScore(batch, encoderStates, context):
    encoderStates = encoderStates
        or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                                                  onmt.utils.Cuda.convert(torch.Tensor()),
                                                                                  { batch.size, self.args.rnnSize })

    score = {}

    self.forwardAndApply(batch, encoderStates, context, def (out, t):
        pred = self.generator.forward(out)
        for b = 1, batch.size:
            if t <= batch.targetSize[b]:
                score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
            
        
    )

    return score

