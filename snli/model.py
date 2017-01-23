import torch
import torch.nn as nn
from torch.autograd import Variable


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.d_embed, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dp_ratio,
                        bidirectional=config.bidirectional)
        self.init_config = self.config.n_cells, self.config.batch_size, self.config.d_hidden
        # self.register_buffer('h0', h0)
        # self.register_buffer('c0', c0)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h0 = Variable(torch.zeros(*self.init_config)).cuda()
        c0 = Variable(torch.zeros(*self.init_config)).cuda()

        _, (hn, _) = self.rnn(inputs, (h0[:, :batch_size].contiguous(), c0[:, :batch_size].contiguous()))
        return hn[-1] if not self.config.bidirectional else hn[-2:].view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.encoder = Encoder(config)
        seq_in_size = 2*config.d_hidden
        if self.config.bidirectional:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2
        self.out = nn.Sequential(
            Linear(*lin_config),
            nn.ReLU(),
            nn.Dropout(p=config.dp_ratio),
            Linear(*lin_config),
            nn.ReLU(),
            Linear(*lin_config),
            nn.ReLU(),
            nn.Dropout(p=config.dp_ratio),
            Linear(seq_in_size, config.d_out))

    def forward(self, batch):
        premise = self.encoder(self.embed(batch.premise))
        hypothesis = self.encoder(self.embed(batch.hypothesis))
        answer = self.out(torch.cat([premise, hypothesis], 1))
        return answer
