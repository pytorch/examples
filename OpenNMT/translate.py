import onmt.init

parser = argparse.ArgumentParser(description='translate.lua')

cmd.option('-config', '', "Read options from this file")

##
## **Data options**
##

cmd.option('-model', required=True, help="Path to model .pt file")
cmd.option('-src',   required=True, help="Source sequence to decode (one line per sequence)")
cmd.option('-tgt',   help="True target sequence (optional)")
cmd.option('-output', default='pred.txt',
           help="Path to output the predictions (each line will be the decoded sequence")

# beam search options
##
## **Beam Search options**
##
cmd.option('-beam_size',       type=int, default=5,   help="Beam size")
cmd.option('-batch_size',      type=int, default=30,  help="Batch size")
cmd.option('-max_sent_length', type=int, default=250, help="Maximum sentence length. If any sequences in srcfile are longer than this then it will error out")
cmd.option('-replace_unk', action="store_true",
           help="""Replace the generated UNK tokens with the source token that
                   had the highest attention weight. If phrase_table is provided,
                   it will lookup the identified source token and give the corresponding
                   target token. If it is not provided (or the identified source token
                   does not exist in the table) then it will copy the source token""")
cmd.option('-phrase_table',
           help="""Path to source-target dictionary to replace UNK
                   tokens. See README.md for the format this file should be in""")
cmd.option('-n_best', type=int, default=1, help="If > 1, it will also output an n_best list of decoded sentences")

##
## **Other options**
##
cmd.option('-gpuid', type=int, default=-1, "ID of the GPU to use (-1 = use CPU, 0 = let cuda choose between available GPUs)")
cmd.option('-fallback_to_cpu', action="store_true", "If = True, fallback to CPU if no GPU available")
cmd.option('-time', action="store_true", "Measure batch translation time")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, PPL: %.4f",
          name, scoreTotal / wordsTotal, math.exp(-scoreTotal/wordsTotal))


def main():
  opt = cmd.parse(arg)

  requiredOptions = [
    "model",
    "src"
  ]

  onmt.utils.Opt.init(opt, requiredOptions)

  srcReader = onmt.utils.FileReader.new(opt.src)
  srcBatch, srcWordsBatch, srcFeaturesBatch = [], [], []

  withGoldScore = opt.tgt.len() > 0

  if withGoldScore:
    tgtReader = onmt.utils.FileReader.new(opt.tgt)
    tgtBatch, tgtWordsBatch, tgtFeaturesBatch = [], [], []

  onmt.translate.Translator.init(opt)

  outFile = io.open(opt.output, 'w')

  sentId, batchId = 1, 1

  predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

  timer = time.time()
  elapsed_time = 0

  while True:
    srcTokens = srcReader.next()
    tgtTokens
    if withGoldScore:
      tgtTokens = tgtReader.next()

    if srcTokens is not None:
      srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
      srcBatch += [srcTokens]
      srcWordsBatch += [srcWords]
      if len(srcFeats) > 0:
        srcFeaturesBatch += [srcFeats]

      if withGoldScore:
        tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)
        tgtBatch += [tgtTokens]
        tgtWordsBatch += [tgtWords]
        if len(tgtFeats) > 0:
          tgtFeaturesBatch += [tgtFeats]

    elif len(srcBatch) == 0:
      break


    if srcTokens == None or len(srcBatch) == opt.batch_size:
      start_time = timer.time()

      predBatch, info = onmt.translate.Translator.translate(srcWordsBatch, srcFeaturesBatch,
                                                            tgtWordsBatch, tgtFeaturesBatch)

      elapsed_time += timer.time() - start_time

      for b in range(predBatch):
        srcSent = " ".join(srcBatch[b])
        predSent = " ".join(predBatch[b])
        outFile.write(predSent + '\n')

        print('SENT ' + sentId + '. ' + srcSent)
        print('PRED ' + sentId + '. ' + predSent)
        print("PRED SCORE. %.4f" % info[b].score)

        predScoreTotal = predScoreTotal + info[b].score
        predWordsTotal = predWordsTotal + len(predBatch[b])

        if withGoldScore:
          tgtSent = " ".join(tgtBatch[b])

          print('GOLD ' + sentId + '. ' + tgtSent)
          print("GOLD SCORE. %.4f" % info[b].goldScore)

          goldScoreTotal = goldScoreTotal + info[b].goldScore
          goldWordsTotal = goldWordsTotal + len(tgtBatch[b])

        if opt.n_best > 1:
          print('\nBEST HYP.')
          for n in range(len(info[b].nBest)):
            nBest = " ".join(info[b].nBest[n].tokens)
            print("[%.4f] %s" % (info[b].nBest[n].score, nBest))

        print('')
        sentId = sentId + 1

      if srcTokens is None:
        break

      batchId = batchId + 1
      srcBatch, srcWordsBatch, srcFeaturesBatch = [], [], []
      if withGoldScore:
        tgtBatch, tgtWordsBatch, tgtFeaturesBatch = [], [], []

  if opt.time:
    sentenceCount = sentId-1
    io.stderr.write("Average sentence translation time per sentence: %g s" %
                    elapsed_time / sentenceCount)

  reportScore('PRED', predScoreTotal, predWordsTotal)

  if withGoldScore:
    reportScore('GOLD', goldScoreTotal, goldWordsTotal)

  outFile.close()

if __name__ == "__main__":
    main()
