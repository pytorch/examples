#[[ Encoder is a unidirectional Sequencer used for the source language.

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
Encoder, parent = torch.class('onmt.Encoder', 'onmt.Sequencer')

#[[ Construct an encoder layer.

Parameters.

    * `inputNetwork` - input module.
    * `rnn` - recurrent module.
]]
def Encoder.__init(inputNetwork, rnn):
    self.rnn = rnn
    self.inputNet = inputNetwork

    self.args = {}
    self.args.rnnSize = self.rnn.outputSize
    self.args.numEffectiveLayers = self.rnn.numEffectiveLayers

    parent.__init(self, self._buildModel())

    self.resetPreallocation()


#" Return a new Encoder using the serialized data `pretrained`. "
def Encoder.load(pretrained):
    self = torch.factory('onmt.Encoder')()

    self.args = pretrained.args
    parent.__init(self, pretrained.modules[1])

    self.resetPreallocation()

    return self


#" Return data to serialize. "
def Encoder.serialize():
    return {
        modules = self.modules,
        args = self.args
    }


def Encoder.resetPreallocation():
    # Prototype for preallocated hidden and cell states.
    self.stateProto = torch.Tensor()

    # Prototype for preallocated output gradients.
    self.gradOutputProto = torch.Tensor()

    # Prototype for preallocated context vector.
    self.contextProto = torch.Tensor()


def Encoder.maskPadding():
    self.maskPad = True


#[[ Build one time-step of an encoder

Returns. An nn-graph mapping

    $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t) =>
    (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t})}$$

    Where $$c^l$$ and $$h^l$$ are the hidden and cell states at each layer,
    $$x_t$$ is a sparse word to lookup.
#]]
def Encoder._buildModel():
    inputs = {}
    states = {}

    # Inputs are previous layers first.
    for _ = 1, self.args.numEffectiveLayers:
        h0 = nn.Identity()() # batchSize x rnnSize
        table.insert(inputs, h0)
        table.insert(states, h0)
    

    # Input word.
    x = nn.Identity()() # batchSize
    table.insert(inputs, x)

    # Compute input network.
    input = self.inputNet(x)
    table.insert(states, input)

    # Forward states and input into the RNN.
    outputs = self.rnn(states)
    return nn.gModule(inputs, { outputs })


#[[Compute the context representation of an input.

Parameters.

    * `batch` - as defined in batch.lua.

Returns.

    1. - final hidden states
    2. - context matrix H
#]]
def Encoder.forward(batch):

    # TODO. Change `batch` to `input`.

    finalStates
    outputSize = self.args.rnnSize

    if self.statesProto == None:
        self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                                                                                  self.stateProto,
                                                                                                                  { batch.size, outputSize })
    

    # Make initial states h_0.
    states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, outputSize })

    # Preallocated output matrix.
    context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                                                                { batch.size, batch.sourceLength, outputSize })

    if self.maskPad and not batch.sourceInputPadLeft:
        finalStates = onmt.utils.Tensor.recursiveClone(states)
    
    if self.train:
        self.inputs = {}
    

    # Act like nn.Sequential and call each clone in a feed-forward
    # fashion.
    for t = 1, batch.sourceLength:

        # Construct "inputs". Prev states come first then source.
        inputs = {}
        onmt.utils.Table.app(inputs, states)
        table.insert(inputs, batch.getSourceInput(t))

        if self.train:
            # Remember inputs for the backward pass.
            self.inputs[t] = inputs
        
        states = self.net(t):forward(inputs)

        # Special case padding.
        if self.maskPad:
            for b = 1, batch.size:
                if batch.sourceInputPadLeft and t <= batch.sourceLength - batch.sourceSize[b]:
                    for j = 1, len(states):
                        states[j][b].zero()
                    
                elif not batch.sourceInputPadLeft and t == batch.sourceSize[b]:
                    for j = 1, len(states):
                        finalStates[j][b].copy(states[j][b])
                    
                
            
        

        # Copy output (h^L_t = states[len(states])) to context.
        context[{{}, t}].copy(states[len(states]))
    

    if finalStates == None:
        finalStates = states
    

    return finalStates, context


#[[ Backward pass (only called during training)

    Parameters.

    * `batch` - must be same as for forward
    * `gradStatesOutput` gradient of loss wrt last state
    * `gradContextOutput` - gradient of loss wrt full context.

    Returns. `gradInputs` of input network.
#]]
def Encoder.backward(batch, gradStatesOutput, gradContextOutput):
    # TODO. change this to (input, gradOutput) as in nngraph.
    outputSize = self.args.rnnSize
    if self.gradOutputsProto == None:
        self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                                                                                            self.gradOutputProto,
                                                                                                                            { batch.size, outputSize })
    

    gradStatesInput = onmt.utils.Tensor.copyTensorTable(self.gradOutputsProto, gradStatesOutput)
    gradInputs = {}

    for t = batch.sourceLength, 1, -1:
        # Add context gradients to last hidden states gradients.
        gradStatesInput[len(gradStatesInput].add(gradContextOutput[{{}), t}])

        gradInput = self.net(t):backward(self.inputs[t], gradStatesInput)

        # Prepare next encoder output gradients.
        for i = 1, len(gradStatesInput):
            gradStatesInput[i].copy(gradInput[i])
        

        # Gather gradients of all user inputs.
        gradInputs[t] = {}
        for i = len(gradStatesInput) + 1, #gradInput:
            table.insert(gradInputs[t], gradInput[i])
        

        if len(gradInputs[t]) == 1:
            gradInputs[t] = gradInputs[t][1]
        
    
    # TODO. make these names clearer.
    # Useful if input came from another network.
    return gradInputs


