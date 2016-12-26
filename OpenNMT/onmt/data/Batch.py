#[[ Return the maxLength, sizes, and non-zero count
    of a batch of `seq`s ignoring `ignore` words.
#]]
def getLength(seq, ignore):
    sizes = torch.IntTensor(len(seq)).zero()
    max = 0
    sum = 0

    for i = 1, len(seq):
        len = seq[i].size(1)
        if ignore != None:
            len = len - ignore
        
        max = math.max(max, len)
        sum = sum + len
        sizes[i] = len
    
    return max, sizes, sum


#[[ Data management and batch creation.

Batch interface reference [size].

    * size. number of sentences in the batch [1]
    * sourceLength. max length in source batch [1]
    * sourceSize.  lengths of each source [batch x 1]
    * sourceInput.  left-padded idx's of source (PPPPPPABCDE) [batch x max]
    * sourceInputFeatures. table of source features sequences
    * sourceInputRev. right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
    * sourceInputRevFeatures. table of reversed source features sequences
    * targetLength. max length in source batch [1]
    * targetSize. lengths of each source [batch x 1]
    * targetNonZeros. number of non-ignored words in batch [1]
    * targetInput. input idx's of target (SABCDEPPPPPP) [batch x max]
    * targetInputFeatures. table of target input features sequences
    * targetOutput. expected output idx's of target (ABCDESPPPPPP) [batch x max]
    * targetOutputFeatures. table of target output features sequences

  TODO. change name of size => maxlen
#]]


#[[ A batch of sentences to translate and targets. Manages padding,
    features, and batch alignment (for efficiency).

    Used by the decoder and encoder objects.
#]]
Batch = torch.class('Batch')

#[[ Create a batch object.

Parameters.

    * `src` - 2D table of source batch indices
    * `srcFeatures` - 2D table of source batch features (opt)
    * `tgt` - 2D table of target batch indices
    * `tgtFeatures` - 2D table of target batch features (opt)
#]]
def Batch.__init(src, srcFeatures, tgt, tgtFeatures):
    src = src or {}
    srcFeatures = srcFeatures or {}
    tgtFeatures = tgtFeatures or {}

    if tgt != None:
        assert(len(src) == #tgt, "source and target must have the same batch size")
    

    self.size = len(src)

    self.sourceLength, self.sourceSize = getLength(src)

    sourceSeq = torch.IntTensor(self.sourceLength, self.size).fill(onmt.Constants.PAD)
    self.sourceInput = sourceSeq.clone()
    self.sourceInputRev = sourceSeq.clone()

    self.sourceInputFeatures = {}
    self.sourceInputRevFeatures = {}

    if len(srcFeatures) > 0:
        for _ = 1, len(srcFeatures[1]):
            table.insert(self.sourceInputFeatures, sourceSeq.clone())
            table.insert(self.sourceInputRevFeatures, sourceSeq.clone())
        
    

    if tgt != None:
        self.targetLength, self.targetSize, self.targetNonZeros = getLength(tgt, 1)

        targetSeq = torch.IntTensor(self.targetLength, self.size).fill(onmt.Constants.PAD)
        self.targetInput = targetSeq.clone()
        self.targetOutput = targetSeq.clone()

        self.targetInputFeatures = {}
        self.targetOutputFeatures = {}

        if len(tgtFeatures) > 0:
            for _ = 1, len(tgtFeatures[1]):
                table.insert(self.targetInputFeatures, targetSeq.clone())
                table.insert(self.targetOutputFeatures, targetSeq.clone())
            
        
    

    for b = 1, self.size:
        sourceOffset = self.sourceLength - self.sourceSize[b] + 1
        sourceInput = src[b]
        sourceInputRev = src[b].index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

        # Source input is left padded [PPPPPPABCDE] .
        self.sourceInput[{{sourceOffset, self.sourceLength}, b}].copy(sourceInput)
        self.sourceInputPadLeft = True

        # Rev source input is right padded [EDCBAPPPPPP] .
        self.sourceInputRev[{{1, self.sourceSize[b]}, b}].copy(sourceInputRev)
        self.sourceInputRevPadLeft = False

        for i = 1, len(self.sourceInputFeatures):
            sourceInputFeatures = srcFeatures[b][i]
            sourceInputRevFeatures = srcFeatures[b][i].index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

            self.sourceInputFeatures[i][{{sourceOffset, self.sourceLength}, b}].copy(sourceInputFeatures)
            self.sourceInputRevFeatures[i][{{1, self.sourceSize[b]}, b}].copy(sourceInputRevFeatures)
        

        if tgt != None:
            # Input. [<s>ABCDE]
            # Ouput. [ABCDE</s>]
            targetLength = tgt[b].size(1) - 1
            targetInput = tgt[b].narrow(1, 1, targetLength)
            targetOutput = tgt[b].narrow(1, 2, targetLength)

            # Target is right padded [<S>ABCDEPPPPPP] .
            self.targetInput[{{1, targetLength}, b}].copy(targetInput)
            self.targetOutput[{{1, targetLength}, b}].copy(targetOutput)

            for i = 1, len(self.targetInputFeatures):
                targetInputFeatures = tgtFeatures[b][i].narrow(1, 1, targetLength)
                targetOutputFeatures = tgtFeatures[b][i].narrow(1, 2, targetLength)

                self.targetInputFeatures[i][{{1, targetLength}, b}].copy(targetInputFeatures)
                self.targetOutputFeatures[i][{{1, targetLength}, b}].copy(targetOutputFeatures)
            
        
    


#[[ Set source input directly,

Parameters.

    * `sourceInput` - a Tensor of size (sequence_length, batch_size, feature_dim)
    ,or a sequence of size (sequence_length, batch_size). Be aware that sourceInput is not cloned here.

#]]
def Batch.setSourceInput(sourceInput):
    assert (sourceInput.dim() >= 2, 'The sourceInput tensor should be of size (seq_len, batch_size, ...)')
    self.size = sourceInput.size(2)
    self.sourceLength = sourceInput.size(1)
    self.sourceInputFeatures = {}
    self.sourceInputRevReatures = {}
    self.sourceInput = sourceInput
    self.sourceInputRev = self.sourceInput.index(1, torch.linspace(self.sourceLength, 1, self.sourceLength):long())
    return self


#[[ Set target input directly.

Parameters.

    * `targetInput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD. Be aware that targetInput is not cloned here.
#]]
def Batch.setTargetInput(targetInput):
    assert (targetInput.dim() == 2, 'The targetInput tensor should be of size (seq_len, batch_size)')
    self.targetInput = targetInput
    self.size = targetInput.size(2)
    self.totalSize = self.size
    self.targetLength = targetInput.size(1)
    self.targetInputFeatures = {}
    self.targetSize = torch.sum(targetInput.transpose(1,2):ne(onmt.Constants.PAD), 2):view(-1):double()
    return self


#[[ Set target output directly.

Parameters.

    * `targetOutput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD.  Be aware that targetOutput is not cloned here.
#]]
def Batch.setTargetOutput(targetOutput):
    assert (targetOutput.dim() == 2, 'The targetOutput tensor should be of size (seq_len, batch_size)')
    self.targetOutput = targetOutput
    self.targetOutputFeatures = {}
    return self



#" Get source input batch at timestep `t`. --"
def Batch.getSourceInput(t):
    # If a regular input, return word id, otherwise a table with features.
    if len(self.sourceInputFeatures) > 0:
        inputs = {self.sourceInput[t]}
        for j = 1, len(self.sourceInputFeatures):
            table.insert(inputs, self.sourceInputFeatures[j][t])
        
        return inputs
    else:
        return self.sourceInput[t]
    


#" Get target input batch at timestep `t`. --"
def Batch.getTargetInput(t):
    # If a regular input, return word id, otherwise a table with features.
    if len(self.targetInputFeatures) > 0:
        inputs = {self.targetInput[t]}
        for j = 1, len(self.targetInputFeatures):
            table.insert(inputs, self.targetInputFeatures[j][t])
        
        return inputs
    else:
        return self.targetInput[t]
    


#" Get target output batch at timestep `t` (values t+1). --"
def Batch.getTargetOutput(t):
    # If a regular input, return word id, otherwise a table with features.
    outputs = { self.targetOutput[t] }
    for j = 1, len(self.targetOutputFeatures):
        table.insert(outputs, self.targetOutputFeatures[j][t])
    
    return outputs


return Batch
