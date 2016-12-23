def _makeFeatEmbedder(opt, dicts):
    return onmt.FeaturesEmbedding(dicts.features,
                                  opt.feat_vec_exponent,
                                  opt.feat_vec_size,
                                  opt.feat_merge)


class Encoder(nn.Container):

    def __init__(self, opt, dicts):
        input_size = opt.word_vec_size
        feat_lut = None
        # Sequences with features.
        if len(dicts.features) > 0:
            feat_lut = _makeFeatEmbedder(opt, dicts)
            inputSize = inputSize + feat_lut.outputSize

        super(Encoder, self).__init__(
            word_lut=nn.LookupTable(dicts.words.size(), opt.word_vec_size)),
            rnn=nn.LSTM(inputSize, opt.rnnSize,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)
        )

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

        self.has_features = feat_lut is not None
        if self.has_features:
            self.add_module('feat_lut', feat_lut)

    def forward(self, input, hidden):
        if self.has_features:
            word_emb = self.word_lut(input[0])
            feat_emb = self.feat_lut(input[1])
            emb = torch.cat([word_emb, feat_emb], 1)
        else:
            emb = self.word_lut(input)

        outputs, next_hidden = self.rnn(input, hidden)
        return outputs, next_hidden

class Decoder(nn.Container):

    def __init__(self, opt, dicts):
        input_size = opt.word_vec_size
        feat_lut = None
        # Sequences with features.
        if len(dicts.features) > 0:
            feat_lut = _makeFeatEmbedder(opt, dicts)
            inputSize = inputSize + feat_lut.outputSize

        super(Decoder, self).__init__(
            word_lut=nn.LookupTable(dicts.words.size(), opt.word_vec_size)),
            rnn=nn.LSTM(inputSize, opt.rnnSize,
                        num_layers=opt.layers,
                        dropout=opt.dropout),
            attn=GlobalAttention(opt.rnnSize),
            dropout=nn.Dropout(opt.dropout)
        )

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)

        self.has_features = feat_lut is not None
        if self.has_features:
            self.add_module('feat_lut', feat_lut)

    def forward(self, input, hidden):
        if self.has_features:
            word_emb = self.word_lut(input[0])
            feat_emb = self.feat_lut(input[1])
            emb = torch.cat([word_emb, feat_emb], 1)
        else:
            emb = self.word_lut(input)

        if self.input_feed:
            emb = torch.cat([emb, input_feed], 1) # 1 step

        outputs, next_hidden = self.rnn(input, hidden)

        attn = self.attn(outputs, context) # FIXME: per timestep?
        attn = self.dropout(attn)
        return attn, next_hidden
