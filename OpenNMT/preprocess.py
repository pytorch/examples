import onmt

import argparse
import os
import torch

parser = argparse.ArgumentParser(description='preprocess.lua')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True, help="Path to the training source data")
parser.add_argument('-train_tgt', required=True, help="Path to the training target data")
parser.add_argument('-valid_src', required=True, help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True, help="Path to the validation target data")

parser.add_argument('-save_data', required=True, help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000, help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000, help="Size of the target vocabulary")
parser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab', help="Path to an existing target vocabulary")
parser.add_argument('-features_vocabs_prefix', help="Path prefix to existing features vocabularies")

parser.add_argument('-seq_length', type=int, default=50,   help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435, help="Random seed")

parser.add_argument('-report_every', type=int, default=100000, help="Report status every this many sentences")

opt = parser.parse_args()


def extract(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split()
    features = None
    data = [token.split('\|') for token in tokens]
    words = [d[0] for d in data if d[0] != '']
    features = [d[1:] for d in data]
    features = list(zip(*features))

    return words, features


def makeVocabulary(filename, size):
    wordVocab = onmt.Dict(
            [onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
             onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD])
    featuresVocabs = []

    with open(filename) as f:
        for sent in f.readlines():
            words, features = extract(sent)
            numFeatures = len(features)
            assert numFeatures == 0, "Features not implemented"

            if len(featuresVocabs) == 0 and numFeatures > 0:
                error("Features not implemented")
                for j in range(numFeatures):
                    featuresVocabs[j] = onmt.Dict(
                            {onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                             onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
            else:
                assert len(featuresVocabs) == numFeatures, (
                    'all sentences must have the same numbers of additional features')

            for i in range(len(words)):
                wordVocab.add(words[i])

                for j in range(numFeatures):
                    featuresVocabs[j].add(features[j][i])

    originalSize = wordVocab.size()
    wordVocab = wordVocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (wordVocab.size(), originalSize))

    return wordVocab, featuresVocabs


def initVocabulary(name, dataFile, vocabFile, vocabSize, featuresVocabsFiles):
    featuresVocabs = []

    wordVocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        wordVocab = onmt.Dict()
        wordVocab.loadFile(vocabFile)
        print('Loaded ' + wordVocab.size() + ' ' + name + ' words')

    if featuresVocabsFiles is not None:
        # If given, discover existing features dictionaries.
        j = 1

        while True:
            file = featuresVocabsFiles + '.' + name + '_feature_' + j + '.dict'

            if not os.path.exists(file):
                break

            print("Reading %s feature %d vocabulary from '%s'..." % (name, j, file))
            featuresVocabs[j] = onmt.Dict()
            featuresVocabs[j].loadFile(file)
            print('Loaded %d labels' % featuresVocabs[j].size())

            j += 1

    if wordVocab is None or (len(featuresVocabs) == 0 and hasFeatures(dataFile)):
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab, genFeaturesVocabs = makeVocabulary(dataFile, vocabSize)

        if wordVocab is None:
            wordVocab = genWordVocab

        if len(featuresVocabs) == 0:
            featuresVocabs = genFeaturesVocabs

    print()
    return {
        'words': wordVocab,
        'features': featuresVocabs
    }


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def saveFeaturesVocabularies(name, vocabs, prefix):
    for j in range(len(vocabs)):
         file = "%s.%s_feature_%d.dict" % (prefix, name, j)
         print("Saving %s feature %d vocabulary to '%s'..." %(name, j, file))
         vocabs[j].writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, srcFeatures = [], []
    tgt, tgtFeatures = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        srcTokens = srcF.readline().split()
        tgtTokens = tgtF.readline().split()

        if not srcTokens or not tgtTokens:
            if srcTokens and not tgtTokens or not srcTokens and tgtTokens:
                print('WARNING. source and target do not have the same number of sentences')
            break

        if len(srcTokens) <= opt.seq_length and len(tgtTokens) <= opt.seq_length:

            srcWords, srcFeats = extract(srcTokens)
            tgtWords, tgtFeats = extract(tgtTokens)

            src += [srcDicts['words'].convertToIdx(srcWords, onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts['words'].convertToIdx(tgtWords, onmt.Constants.UNK_WORD,
                        onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD)]

            if len(srcDicts['features']) > 0:
                srcFeatures += [generateSourceFeatures(srcDicts['features'], srcFeats)]

            if len(tgtDicts['features']) > 0:
               tgtFeatures += [generateTargetFeatures(tgtDicts['features'], tgtFeats)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

        if len(srcDicts['features']) > 0:
            srcFeatures = [srcFeatures[idx] for idx in perm]
        if len(tgtDicts['features']) > 0:
            tgtFeatures = [tgtFeatures[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    if len(srcDicts['features']) > 0:
        srcFeatures = [srcFeatures[idx] for idx in perm]
    if len(tgtDicts['features']) > 0:
        tgtFeatures = [tgtFeatures[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    srcData = {
        'words': src,
        'features': srcFeatures
    }

    tgtData = {
        'words': tgt,
        'features': tgtFeatures
    }

    return srcData, tgtData


def main():

    dicts = {}
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size, opt.features_vocabs_prefix)
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size, opt.features_vocabs_prefix)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                    dicts['src'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src']['words'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt']['words'], opt.save_data + '.tgt.dict')

    if opt.features_vocabs_prefix is None:
        saveFeaturesVocabularies('source', dicts['src']['features'], opt.save_data)
        saveFeaturesVocabularies('target', dicts['tgt']['features'], opt.save_data)


    print('Saving data to \'' + opt.save_data + '-train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '-train.pt')


if __name__ == "__main__":
    main()
