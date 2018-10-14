import logging

import torch
import torch.nn as nn
import numpy as np

SOS_TOKEN = '~'
PAD_TOKEN = '#'

def sequences_to_tensors(sequences, token_to_idx):
    """ Casts a list of sequences into rnn-digestable padded tensor """
    seq_idx = []
    for seq in sequences:
        seq_idx.append([token_to_idx[token] for token in seq])
    sequences = [torch.LongTensor(x) for x in seq_idx]
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=token_to_idx[PAD_TOKEN])

def load_dataset(file_name='names'):
    with open(file_name) as file:
        sequences = file.read()[:-1].split('\n')
        sequences = [SOS_TOKEN + seq.lower() for seq in sequences]

    logging.info('number of sequences: {}'.format(len(sequences)))
    for seq in sequences[::1000]:
        print(seq[1:].capitalize())

    MAX_LENGTH = max(map(len, sequences))
    logging.info('max length: {}'.format(MAX_LENGTH))

    idx_to_token = list(set([token for seq in sequences for token in seq]))
    idx_to_token.append(PAD_TOKEN)
    n_tokens = len(idx_to_token)
    logging.info('number of unique tokens: {}'.format(n_tokens))

    token_to_idx = {token: idx_to_token.index(token) for token in idx_to_token}
    assert len(idx_to_token) ==  len(token_to_idx), 'dicts must have same lenghts'

    logging.debug('processing tokens')
    sequences = sequences_to_tensors(sequences, token_to_idx)
    return sequences, token_to_idx, idx_to_token

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(inputs.size(0))
    for start_idx in range(0, inputs.size(0) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]