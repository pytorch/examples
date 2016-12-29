import onmt
from torch.autograd import Variable


class Dataset(object):
    # FIXME: randomize
    def __init__(self, srcData, tgtData, batchSize, cuda):
        self.src = srcData['words']
        self.tgt = tgtData['words']
        self.cuda = cuda
        # FIXME
        # self.srcFeatures = srcData.features
        # self.tgtFeatures = tgtData.features
        assert(len(self.src) == len(self.tgt))
        self.batchSize = batchSize
        self.numBatches = len(self.src) // batchSize

    def _batchify(self, data):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            out[i].narrow(0, 0, data[i].size(0)).copy_(data[i])
        return Variable(out)

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize])
        tgtBatch = self._batchify(
            self.tgt[index*self.batchSize:(index+1)*self.batchSize])

        if self.cuda:
            srcBatch = srcBatch.cuda()
            tgtBatch = tgtBatch.cuda()

        # FIXME
        srcBatch = srcBatch.t().contiguous()
        tgtBatch = tgtBatch.t().contiguous()

        return srcBatch, tgtBatch

    def __len__(self):
        return self.numBatches
