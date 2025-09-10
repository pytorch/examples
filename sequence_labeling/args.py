import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--emb_path', default='.data/glove.6B.100d.txt', type=str, help='path to pretrained embeddings')
parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam', 'nesterov'],
                    help='optimizer to use')
parser.add_argument('--sgd_momentum', default=0.9, type=float, help='momentum for stochastic gradient descent')
parser.add_argument('--rnn', default='lstm', type=str, choices=['lstm', 'gru'], help='RNN type')
parser.add_argument('--rnn_dim', default=100, type=int, help='RNN hidden state dimension')
parser.add_argument('--word_emb_dim', default=100, type=int, help='word embedding dimension')
parser.add_argument('--char_emb_dim', default=50, type=int, help='char embedding dimension')
parser.add_argument('--char_rnn_dim', default=50, type=int, help='char RNN hidden state dimension')
parser.add_argument('--word_min_freq', default=2, type=int, help='minimum frequency threshold in word vocabulary')
parser.add_argument('--train_batch_size', default=16, type=int, help='batch size for training phase')
parser.add_argument('--val_batch_size', default=64, type=int, help='batch size for evaluation phase')
parser.add_argument('--early_stopping_patience', default=30, type=int, help='early stopping patience')
parser.add_argument('--dropout_before_rnn', default=0.5, type=float, help='dropout rate on RNN inputs')
parser.add_argument('--dropout_after_rnn', default=0.5, type=float, help='dropout rate on RNN outputs')
parser.add_argument('--lr', default=0.01, type=float, help='starting learning rate')
parser.add_argument('--lr_decay', default=0.001, type=float, help='learning rate decay factor')
parser.add_argument('--min_lr', default=0.0005, type=float, help='minimum learning rate')
parser.add_argument('--lr_shrink', default=0.5, type=float, help='learning rate reducing factor')
parser.add_argument('--lr_shrink_patience', default=0, type=float, help='learning rate reducing patience')
parser.add_argument('--crf', default='small', type=str, choices=['none', 'small', 'large'], help='CRF type or no CRF')
parser.add_argument('--max_epochs', default=300, type=int, help='maximum training epochs')

args = parser.parse_args()

args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
