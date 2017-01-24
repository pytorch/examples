import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_embed', type=int, default=10)
    parser.add_argument('--d_hidden', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--val_every', type=int, default=1000000)
    parser.add_argument('--save_every', type=int, default=100000)
    parser.add_argument('--dp_ratio', type=int, default=0.0)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    args = parser.parse_args()
    return args
