import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules


def _makeFeatEmbedder(opt, dicts):
    return onmt.FeaturesEmbedding(dicts['features'],
                                  opt.feat_vec_exponent,
                                  opt.feat_vec_size,
                                  opt.feat_merge)


class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        inputSize = opt.word_vec_size
        feat_lut = None
        # Sequences with features.
        if len(dicts['features']) > 0:
            feat_lut = _makeFeatEmbedder(opt, dicts)
            inputSize = inputSize + feat_lut.outputSize

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts['words'].size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)


        # self.rnn.bias_ih_l0.data.div_(2)
        # self.rnn.bias_hh_l0.data.copy_(self.rnn.bias_ih_l0.data)

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

        self.has_features = feat_lut is not None
        if self.has_features:
            self.add_module('feat_lut', feat_lut)

    def forward(self, input):
        if self.has_features:
            word_emb = self.word_lut(input[0])
            feat_emb = self.feat_lut(input[1])
            emb = torch.cat([word_emb, feat_emb], 1)
        else:
            emb = self.word_lut(input)

        batch_size = emb.size(1)
        h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
        h_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
        c_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
        hidden_0 = (h_0, c_0)
        outputs, hidden_t = self.rnn(emb, hidden_0)
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)


        self.layers = []
        for i in range(num_layers):
            layer = nn.LSTMCell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i != len(self.layers):
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        feat_lut = None
        # Sequences with features.
        if len(dicts['features']) > 0:
            feat_lut = _makeFeatEmbedder(opt, dicts)
            input_size = input_size + feat_lut.outputSize

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts['words'].size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        # self.rnn.bias_ih.data.div_(2)
        # self.rnn.bias_hh.data.copy_(self.rnn.bias_ih.data)

        self.hidden_size = opt.rnn_size

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)

        self.has_features = feat_lut is not None
        if self.has_features:
            self.add_module('feat_lut', feat_lut)

    def forward(self, input, hidden, context):
        if self.has_features:
            word_emb = self.word_lut(input[0])
            feat_emb = self.feat_lut(input[1])
            emb = torch.cat([word_emb, feat_emb], 1)
        else:
            emb = self.word_lut(input)

        batch_size = input.size(1)

        h_size = (batch_size, self.hidden_size)
        output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

        outputs = []
        for emb_t in emb.chunk(emb.size(0)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.generate = False

    def set_generate(self, enabled):
        self.generate = enabled

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        out = self.decoder(tgt, enc_hidden, context)
        if self.generate:
            out = self.generator(out)

        return out
