########################################
# Data Fetching Script for PTB
########################################

import torch
import os.path

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def addword(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        
        return self.word2idx[word]

    def ntokens(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dic = Dictionary()
        self.train=self._loadfile(os.path.join(path, 'train.txt'))
        self.valid=self._loadfile(os.path.join(path, 'valid.txt'))
        self.test =self._loadfile(os.path.join(path, 'test.txt'))

    # | Tokenize a text file.
    def _loadfile(self, path):
        # Read words from file.
        assert(os.path.exists(path))
        tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dic.addword(word)
                    tokens += 1
    
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dic.word2idx[word]
                    token += 1
    
        # Final dataset.
        return ids
