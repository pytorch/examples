import onmt
import torch
from torch.autograd import Variable


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)
        self.model = checkpoint['model']

        self.model.eval()

        if opt.cuda:
            self.model.cuda()
        else:
            self.model.cpu()

        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData,
            self.opt.batch_size, self.opt.cuda)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    # FIXME phrase table
                    tokens[i] = src[maxIndex[0]]

        return tokens

    def translateBatch(self, batch):
        srcBatch, tgtBatch = batch
        batchSize = srcBatch.size(0)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src

        # have to execute the encoder manually to deal with padding
        encStates = None
        context = []
        for srcBatch_t in srcBatch.chunk(srcBatch.size(1), dim=1):
            encStates, context_t = self.model.encoder(srcBatch_t, hidden=encStates)
            batchPadIdx = srcBatch_t.data.squeeze(1).eq(onmt.Constants.PAD).nonzero()
            if batchPadIdx.nelement() > 0:
                batchPadIdx = batchPadIdx.squeeze(1)
                encStates[0].data.index_fill_(1, batchPadIdx, 0)
                encStates[1].data.index_fill_(1, batchPadIdx, 0)
            context += [context_t]

        encStates = (self.model._fix_enc_hidden(encStates[0]),
                      self.model._fix_enc_hidden(encStates[1]))

        context = torch.cat(context)
        rnnSize = context.size(2)

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(onmt.Constants.PAD)
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention):
                m.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()
        if tgtBatch is not None:
            decStates = encStates
            decOut = self.model.make_init_decoder_output(context)
            self.model.decoder.apply(applyContextMask)
            initOutput = self.model.make_init_decoder_output(context)

            decOut, decStates, attn = self.model.decoder(
                    tgtBatch[:, :-1], decStates, context, initOutput)
            for dec_t, tgt_t in zip(decOut.transpose(0, 1), tgtBatch.transpose(0, 1)[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))
        decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                     Variable(encStates[1].data.repeat(1, beamSize, 1)))

        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = srcBatch.data.eq(onmt.Constants.PAD).unsqueeze(0).repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):

            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                               if not b.done]).t().contiguous().view(1, -1)

            decOut, decStates, attn = self.model.decoder(
                Variable(input).transpose(0, 1), decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.transpose(0, 1).squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                    .view(*newSize))

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.transpose(0, 1).data[:, b].ne(onmt.Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]

        return allHyp, allScores, allAttn, goldScores

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset[0]
        batch = [x.transpose(0, 1) for x in batch]

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(batch)

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batch[0].size(0)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                        for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore
