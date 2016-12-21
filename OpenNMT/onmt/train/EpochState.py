#[[ Class for managing the training process by logging and storing
    the state of the current epoch.
]]
EpochState = torch.class("EpochState")

# Initialize for epoch `epoch` and training `status` (current loss)
def EpochState.__init(epoch, numIterations, learningRate, lastValidPpl, status):
    self.epoch = epoch
    self.numIterations = numIterations
    self.learningRate = learningRate
    self.lastValidPpl = lastValidPpl

    if status != None:
        self.status = status
    else:
        self.status = {}
        self.status.trainNonzeros = 0
        self.status.trainLoss = 0


    self.timer = torch.Timer()
    self.numWordsSource = 0
    self.numWordsTarget = 0

    self.minFreeMemory = 100000000000


# Update training status. Takes `batch` (described in data.lua) and last losses.
def EpochState.update(batches, losses):
    for i = 1,len(batches):
        self.numWordsSource = self.numWordsSource + batches[i].size * batches[i].sourceLength
        self.numWordsTarget = self.numWordsTarget + batches[i].size * batches[i].targetLength
        self.status.trainLoss = self.status.trainLoss + losses[i]
        self.status.trainNonzeros = self.status.trainNonzeros + batches[i].targetNonZeros



# Log to status stdout.
def EpochState.log(batchIndex, json):
    if json:
        freeMemory = onmt.utils.Cuda.freeMemory()
        if freeMemory < self.minFreeMemory:
            self.minFreeMemory = freeMemory


        obj = {
            time = os.time(),
            epoch = self.epoch,
            iteration = batchIndex,
            totalIterations = self.numIterations,
            learningRate = self.learningRate,
            trainingPerplexity = self.getTrainPpl(),
            freeMemory = freeMemory,
            lastValidationPerplexity = self.lastValidPpl,
            processedTokens = {
                source = self.numWordsSource,
                target = self.numWordsTarget
            }
        }

        onmt.utils.Log.logJson(obj)
    else:
        timeTaken = self.getTime()

        stats = ''
        stats = stats + string.format('Epoch %d ; ', self.epoch)
        stats = stats + string.format('Iteration %d/%d ; ', batchIndex, self.numIterations)
        stats = stats + string.format('Learning rate %.4f ; ', self.learningRate)
        stats = stats + string.format('Source tokens/s %d ; ', self.numWordsSource / timeTaken)
        stats = stats + string.format('Perplexity %.2f', self.getTrainPpl())
        print(stats)



def EpochState.getTrainPpl():
    return math.exp(self.status.trainLoss / self.status.trainNonzeros)


def EpochState.getTime():
    return self.timer.time().real


def EpochState.getStatus():
    return self.status


def EpochState.getMinFreememory():
    return self.minFreeMemory


return EpochState
