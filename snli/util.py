import os
from argparse import ArgumentParser

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of total epochs to run.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size. (default: 128)')
    parser.add_argument('--d_embed', type=int, default=100,
                        help='the size of each embedding vector.')
    parser.add_argument('--d_proj', type=int, default=300,
                        help='the size of each projection layer.')
    parser.add_argument('--d_hidden', type=int, default=300,
                        help='the number of features in the hidden state.')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='the number of recurrent layers. (default: 50)')
    parser.add_argument('--log_every', type=int, default=50,
                        help='iteration period to output log.')
    parser.add_argument('--lr',type=float, default=.001,
                        help='initial learning rate.')
    parser.add_argument('--dev_every', type=int, default=1000,
                        help='log period of validation results.')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='model checkpoint period.')
    parser.add_argument('--dp_ratio', type=int, default=0.2,
                        help='probability of an element to be zeroed.')
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn',
                        help='disable bidirectional LSTM.')
    parser.add_argument('--preserve-case', action='store_false', dest='lower',
                        help='case-sensitivity.')
    parser.add_argument('--no-projection', action='store_false', dest='projection',
                        help='disable projection layer.')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb',
                        help='enable embedding word training.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id to use. (default: 0)')
    parser.add_argument('--save_path', type=str, default='results',
                        help='save path of results.')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'),
                        help='name of vector cache directory, which saved input word-vectors.')
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d',
                        help='one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes.'
                        'Alternatively, one of or a list of available pretrained vectors: '
                        'charngram.100d fasttext.en.300d fasttext.simple.300d'
                        'glove.42B.300d glove.840B.300d glove.twitter.27B.25d'
                        'glove.twitter.27B.50d glove.twitter.27B.100d glove.twitter.27B.200d'
                        'glove.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d')
    parser.add_argument('--resume_snapshot', type=str, default='',
                        help='model snapshot to resume.')
    parser.add_argument('--dry-run', action='store_true',
                        help='run only a few iterations')
    args = parser.parse_args()
    return args
