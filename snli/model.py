import torch
import torch.nn as nn
from torch.autograd import Variable

from spinn import SPINN

class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class BatchNorm(Bottle, nn.BatchNorm1d):
    pass


class Feature(nn.Module):

    def __init__(self, size, dropout):
        super(Feature, self).__init__()
        self.bn = nn.BatchNorm1d(size * 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prem, hypo):
        return self.dropout(self.bn(torch.cat(
            [prem, hypo, prem - hypo, prem * hypo], 1)))


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.rnn_dropout,
                           bidirectional=config.birnn)

    def forward(self, inputs, _):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.embed_bn = BatchNorm(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)
        self.encoder = SPINN(config) if config.spinn else Encoder(config)
        feat_in_size = config.d_hidden * (
            2 if self.config.birnn and not self.config.spinn else 1)
        self.feature = Feature(feat_in_size, config.mlp_dropout)
        self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)
        self.relu = nn.ReLU()
        mlp_in_size = 4 * feat_in_size
        mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu,
               nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        for i in range(config.n_mlp_layers - 1):
            mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu,
                        nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        mlp.append(nn.Linear(config.d_mlp, config.d_out))
        self.out = nn.Sequential(*mlp)

    def forward(self, batch):
        # import pdb
        # pdb.set_trace()
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        if self.config.fix_emb:
            prem_embed = Variable(prem_embed.data)
            hypo_embed = Variable(hypo_embed.data)
        if self.config.projection:
            prem_embed = self.projection(prem_embed)  # no relu
            hypo_embed = self.projection(hypo_embed)
        prem_embed = self.embed_dropout(self.embed_bn(prem_embed))
        hypo_embed = self.embed_dropout(self.embed_bn(hypo_embed))
        if hasattr(batch, 'premise_transitions'):
            prem_trans = batch.premise_transitions
            hypo_trans = batch.hypothesis_transitions
        else:
            prem_trans = hypo_trans = None
        premise = self.encoder(prem_embed, prem_trans)
        hypothesis = self.encoder(hypo_embed, hypo_trans)
        scores = self.out(self.feature(premise, hypothesis))
        #print(premise[0][:5], hypothesis[0][:5])
        return scores
