class ONMTDataset(object):

    def __init__(self, srcData, tgtData, batchSize):
        self.src = srcData.words
        self.tgt = tgtData.words
        # self.srcFeatures = srcData.features
        # self.tgtFeatures = tgtData.features
        assert(len(self.src) == len(self.tgt))
        self.batchSize = batchSize
        self.numBatches = len(self.src) // batchSize

    def __getitem__(self, index):
        assert(index < self.numBatches)
        srcBatch = self.src[index * self.batchSize:index * (self.batchSize+1)]
        tgtBatch = self.tgt[index * self.batchSize:index * (self.batchSize+1)]

        return srcBatch, tgtBatch

    def __len__(self):
        return self.numBatches
