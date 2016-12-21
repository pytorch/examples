# Class for saving and loading models during training.
Checkpoint = torch.class("Checkpoint")

def Checkpoint.__init(options, model, optim, dataset):
    self.options = options
    self.model = model
    self.optim = optim
    self.dataset = dataset

    self.savePath = self.options.save_model


def Checkpoint.save(filePath, info):
    info.learningRate = self.optim.getLearningRate()
    info.optimStates = self.optim.getStates()

    data = {
        models = {},
        options = self.options,
        info = info,
        dicts = self.dataset.dicts
    }

    for k, v in pairs(self.model):
        data.models[k] = v.serialize()


    torch.save(filePath, data)


#" Save the model and data in the middle of an epoch sorting the iteration. "
def Checkpoint.saveIteration(iteration, epochState, batchOrder, verbose):
    info = {}
    info.iteration = iteration + 1
    info.epoch = epochState.epoch
    info.epochStatus = epochState.getStatus()
    info.batchOrder = batchOrder

    filePath = string.format('%s_checkpoint.t7', self.savePath)

    if verbose:
        print('Saving checkpoint to \'' + filePath + '\'...')


    # Succeed serialization before overriding existing file
    self.save(filePath + '.tmp', info)
    os.rename(filePath + '.tmp', filePath)


def Checkpoint.saveEpoch(validPpl, epochState, verbose):
    info = {}
    info.validPpl = validPpl
    info.epoch = epochState.epoch + 1
    info.iteration = 1
    info.trainTimeInMinute = epochState.getTime() / 60

    filePath = string.format('%s_epoch%d_%.2f.t7', self.savePath, epochState.epoch, validPpl)

    if verbose:
        print('Saving checkpoint to \'' + filePath + '\'...')


    self.save(filePath, info)


return Checkpoint
