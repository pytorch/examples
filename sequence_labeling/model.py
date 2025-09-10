import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rnn(input_dim, hidden_dim, rnn_type):
    if rnn_type == 'gru':
        return nn.GRU(input_dim, hidden_dim, bidirectional=True)
    elif rnn_type == 'lstm':
        return nn.LSTM(input_dim, hidden_dim, bidirectional=True)
    else:
        raise Exception('Unknown RNN type')


class SequenceLabelingModel(nn.Module):
    def __init__(self, args, logger):
        super(SequenceLabelingModel, self).__init__()

        self.args = args
        self.logger = logger

        self.word_emb = WordEmbedding(args, logger)
        self.word_emb.requires_grad = False

        if args.char_rnn_dim > 0 and args.char_emb_dim > 0:
            self.char_emb = nn.Embedding(len(args.char2idx), args.char_emb_dim)
            self.char_rnn = get_rnn(args.char_emb_dim, args.char_rnn_dim, args.rnn)
            self.rnn = get_rnn(args.word_emb_dim + args.char_rnn_dim * 2, args.rnn_dim, args.rnn)
        else:
            self.rnn = get_rnn(args.word_emb_dim, args.rnn_dim, args.rnn)
            self.char_emb = None
            self.char_rnn = None

        self.dropout_before_rnn = nn.Dropout(args.dropout_before_rnn)
        self.dropout_after_rnn = nn.Dropout(args.dropout_after_rnn)

        if args.crf in {'small', 'large'}:
            self.loss = CRF(args.rnn_dim * 2, len(args.tag2idx), args.tag2idx[args.tag_bos], args.tag2idx[args.tag_eos],
                            args.tag2idx[args.tag_pad], large_model=args.crf == 'large')
        else:
            self.loss = CELoss(args.rnn_dim * 2, len(args.tag2idx))

    def _get_char_emb(self, chars):
        """
        :param chars: batch_size x seq_len x word_len
        :return: seq_len x batch_size x char_emb_dim
        """
        chars, chars_lens = chars[0], chars[2]

        batch_size, seq_len, word_len = chars.size()
        n = seq_len * batch_size
        chars = chars.permute(2, 1, 0).contiguous().view(word_len, n)
        chars_lens = chars_lens.t().contiguous().view(1, n).expand(word_len, n)
        mask = torch.range(0, n - 1).long().view(1, n).expand(word_len, n).cuda() < chars_lens

        char_embeds = self.char_rnn(self.char_emb(chars))[0]
        mask = mask.view(word_len, n, 1).expand_as(char_embeds)
        return (char_embeds * mask.float()).max(0)[0].view(seq_len, batch_size, -1)

    def _get_rnn_features(self, rnn, x, lengths):
        """
        :param x: seq_len x batch_size x emb_dim
        :param lengths: batch_size x 1
        :return: seq_len x batch_size x hidden_dim * 2
        """
        lengths, idx_sort = lengths.sort(descending=True)
        _, idx_unsort = idx_sort.sort(descending=False)
        emb = x.index_select(1, idx_sort)
        emb_packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.tolist())
        out = rnn(emb_packed)[0]
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        return out.index_select(1, idx_unsort)

    def _get_features(self, batch):
        """
        batch.word[0]: seq_len x batch_size
        batch.word[1]: batch_size
        batch.char[0]: batch_size x seq_len x word_len
        :return: seq_len x batch_size x rnn_dim * 2
        """
        words, seq_lens = batch.word[0], batch.word[1]
        embeds = self.word_emb.forward(words)

        if self.args.char_rnn_dim > 0 and self.args.char_emb_dim > 0:
            char_embeds = self._get_char_emb(batch.char)
            embeds = torch.cat([embeds, char_embeds], 2)

        embeds = self.dropout_before_rnn(embeds)
        features = self._get_rnn_features(self.rnn, embeds, seq_lens)
        return self.dropout_after_rnn(features)

    def forward(self, batch):
        """
        batch.word[0]: seq_len x batch_size
        batch.word[1]: batch_size
        batch.char[0]: batch_size x seq_len x word_len
        :return: scalar tensor
        """
        seq_lens = batch.word[1]
        features = self._get_features(batch)
        return self.loss.forward(features, batch.tag, seq_lens)

    def decode(self, batch):
        """
        batch.word[0]: seq_len x batch_size
        batch.word[1]: batch_size
        batch.char[0]: batch_size x seq_len x word_len
        :return: seq_len x batch_size
        """
        seq_lens = batch.word[1]
        features = self._get_features(batch)
        return self.loss.decode(features, seq_lens)


class WordEmbedding(nn.Module):
    def __init__(self, args, logger):
        super(WordEmbedding, self).__init__()

        self.lut = nn.Embedding(len(args.word2idx), args.word_emb_dim)
        self.lut.weight.data.uniform_(-0.1, 0.1)

        logger.info('Loading word embeds')
        word_embeds = {}
        for line in codecs.open(args.emb_path, 'r', 'utf-8'):
            line = line.strip().split()
            if len(line) != args.word_emb_dim + 1:
                continue
            word_embeds[line[0]] = torch.Tensor([float(i) for i in line[1:]])

        logger.info('Matching word embeds')
        count_raw, count_lower = 0, 0
        for word, idx in args.word2idx.items():
            if word in word_embeds:
                self.lut.weight.data[idx].copy_(word_embeds[word])
                count_raw += 1
            elif word.lower() in word_embeds:
                self.lut.weight.data[idx].copy_(word_embeds[word.lower()])
                count_lower += 1

        logger.info('Coverage %.4f (%d+%d/%d, raw+lower)' % (float(count_raw + count_lower) / len(args.word2idx),
                                                             count_raw, count_lower, len(args.word2idx)))

    def forward(self, words):
        """
        :param words: seq_len x batch_size
        :return: seq_len x batch_size x emb_dim
        """
        return self.lut.forward(words)


class CRF(nn.Module):
    def __init__(self, feature_dim, tags_num, bos_ix, eos_ix, pad_ix, large_model=False):
        super(CRF, self).__init__()
        self.tags_num = tags_num
        self.large_model = large_model
        self.bos_ix, self.eos_ix, self.pad_ix = bos_ix, eos_ix, pad_ix

        if self.large_model:
            self.feat2tag = nn.Linear(feature_dim, self.tags_num * self.tags_num)
        else:
            self.feat2tag = nn.Linear(feature_dim, self.tags_num)
            self.transitions = nn.Parameter(torch.zeros(self.tags_num, self.tags_num))

    def _get_crf_scores(self, features):
        """
        :param features: seq_len x batch_size x feature_dim
        :return: seq_len x batch_size x tags_num x tags_num
        """
        s_len, b_size, n_tags = features.size(0), features.size(1), self.tags_num
        if self.large_model:
            return self.feat2tag(features).view(s_len, b_size, n_tags, n_tags)
        else:
            emit_scores = self.feat2tag(features).view(s_len, b_size, n_tags, 1).expand(s_len, b_size, n_tags, n_tags)
            transition_scores = self.transitions.view(1, 1, n_tags, n_tags).expand(s_len, b_size, n_tags, n_tags)
            return emit_scores + transition_scores

    def _get_gold_scores(self, crf_scores, tags, seq_lens):
        """
        :param crf_scores: seq_len x batch_size x tags_num x tags_num
        :param tags: seq_len x batch_size
        :param seq_lens: batch_size
        :return: scalar tensor
        """
        s_len, b_size = crf_scores.size(0), crf_scores.size(1)
        pad_tags = torch.Tensor(1, b_size).long().fill_(self.pad_ix).cuda()
        bigram_tags = self.tags_num * tags + torch.cat([tags[1:, :], pad_tags], 0)

        gold_score = crf_scores.view(s_len, b_size, -1).gather(2, bigram_tags.view(s_len, b_size, 1)).squeeze(2)
        gold_score = gold_score.cumsum(0).gather(0, seq_lens.view(1, b_size) - 1).sum()
        return gold_score

    def forward(self, features, tags, seq_lens):
        """
        :param features: seq_len x batch_size x feature_dim
        :param tags: seq_len x batch_size
        :param seq_lens: batch_size
        :return: scalar tensor
        """
        s_len, b_size, n_tags = features.size(0), features.size(1), self.tags_num
        crf_scores = self._get_crf_scores(features)
        gold_scores = self._get_gold_scores(crf_scores, tags, seq_lens)

        cur_score = crf_scores[0, :, self.bos_ix, :].contiguous()
        for idx in range(1, s_len):
            next_score = cur_score.view(b_size, n_tags, 1).expand(b_size, n_tags, n_tags) + crf_scores[idx, :, :, :]
            next_score = self.log_sum_exp(next_score)
            mask = (idx < seq_lens).view(b_size, 1).expand_as(next_score).float()
            cur_score = mask * next_score + (1 - mask) * cur_score
        cur_score = cur_score[:, self.eos_ix].sum()

        return cur_score - gold_scores

    def decode(self, features, seq_lens):
        """
        :param features: seq_len x batch_size x feature_dim
        :param seq_lens: batch_size
        :return: seq_len x batch_size
        """
        s_len, b_size, n_tags = features.size(0), features.size(1), self.tags_num
        crf_scores = self._get_crf_scores(features)

        cur_score = crf_scores[0, :, self.bos_ix, :].contiguous()
        back_pointers = []
        for idx in range(1, s_len):
            next_score = cur_score.view(b_size, n_tags, 1).expand(b_size, n_tags, n_tags) + crf_scores[idx, :, :, :]
            cur_score, cur_ptr = next_score.max(1)
            cur_ptr.masked_fill_((idx >= seq_lens).view(b_size, 1).expand_as(cur_ptr), self.eos_ix)
            back_pointers.append(cur_ptr)

        best_seq = torch.Tensor(s_len, b_size).long().fill_(self.eos_ix).cuda()
        best_seq[0, :] = self.bos_ix
        ptr = torch.Tensor(b_size, 1).long().fill_(self.eos_ix).cuda()
        for idx in range(s_len - 2, -1, -1):
            ptr = back_pointers[idx].gather(1, ptr)
            best_seq[idx+1, :] = ptr.squeeze(1)
        return best_seq

    @staticmethod
    def log_sum_exp(v):
        """
        :param v: (p, q, r) Variable
        :return: (p, r) Variable
        """
        p, r = v.size(0), v.size(2)
        max_v = v.max(1)[0]
        return max_v + (v - max_v.view(p, 1, r).expand_as(v)).exp().sum(1).log()


class CELoss(nn.Module):
    def __init__(self, feature_dim, tags_num):
        super(CELoss, self).__init__()
        self.hidden2tag = nn.Linear(feature_dim, tags_num)
        self.tags_num = tags_num

    def _get_scores(self, features):
        """
        :param features: seq_len x batch_size x feature_dim
        :return: seq_len x batch_size x tags_num
        """
        return F.log_softmax(self.hidden2tag(features), dim=2)

    def forward(self, features, tags, seq_lens):
        """
        :param features: seq_len x batch_size x feature_dim
        :param tags: seq_len x batch_size
        :param seq_lens: batch_size
        :return: scalar tensor
        """
        s_len, b_size = features.size(0), features.size(1)
        loss = self._get_scores(features).gather(2, tags.view(s_len, b_size, 1)).squeeze(2)
        loss = loss.cumsum(0).gather(0, seq_lens.view(1, b_size) - 1).sum()
        return -loss

    def decode(self, features, seq_lens):
        """
        :param features: seq_len x batch_size x feature_dim
        :param seq_lens: batch_size
        :return: seq_len x batch_size
        """
        return self._get_scores(features).max(2)[1]
