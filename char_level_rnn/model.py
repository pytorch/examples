import torch
import torch.nn as nn
import torch.nn.functional as F

def glorot_normal_initializer(m):
    """ Applies Glorot Normal initialization to layer parameters.
    
    "Understanding the difficulty of training deep feedforward neural networks" 
    by Glorot, X. & Bengio, Y. (2010)

    Args:
        m (nn.Module): a particular layer whose params are to be initialized.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

class CharRNN(nn.Module):
    """ Character-level recurrent neural network.

    This module imitates the basic structure and flow of an 
    RNN with embeddings, hidden states and output softmax.
    Alternatively you can use `LSTM` and `GRU` recurrent modules.

    Args:
        n_tokens (int): number of unique tokens in corpus.
        emb_size (int): dimensionality of each embedding.
        hidden_size (int): number of hidden units in RNN hidden layer.
        pad_id (int): token_id of the padding token.
    """

    def __init__(self, num_layers, rnn_type, n_tokens, emb_size, hidden_size, dropout, pad_id):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.upper()

        self.embedding = nn.Embedding(n_tokens, emb_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        if self.rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(input_size=emb_size, hidden_size=hidden_size, 
                                                num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            raise UserWarning('Unknown RNN type.')
        self.logits = nn.Linear(hidden_size, n_tokens).apply(glorot_normal_initializer)

    def forward(self, input_step, hidden):
        """ Implements the forward pass of the char-level RNN.

        Args:
            inputs (torch.LongTensor): input step token batch to feed the network.
            hidden (torch.Tensor): hidden states of the RNN from previous time-step.
        Returns:
            torch.Tensor: output log softmax probability distribution over tokens.
            torch.Tensor: hidden states of the RNN from current time-step.
        """
        embedded = self.embedding(input_step)
        embedded = self.dropout(embedded)
        outputs, hidden = self.rnn(embedded.view(input_step.size(0), 1, -1), hidden)
        logits = self.logits(outputs.view(-1, self.hidden_size))
        probs = F.log_softmax(logits, dim=1)
        return probs, hidden

    def initHidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
