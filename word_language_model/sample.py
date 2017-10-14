import torch
import data
from torch.autograd import Variable
import numpy as np

torch.set_printoptions(threshold=15000)
LT = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
args_bptt = 1
eval_batch_size = 1
corpus = data.Corpus("data/linux")
vocab_size = len(corpus.dictionary) # aka notokens

# Load the best saved model.
with open("model.pt", 'rb') as f:
    model = torch.load(f)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def sample_fn():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.rnn.flatten_parameters()
    predicted_idxs = []
    hidden = model.init_hidden(eval_batch_size)
    data = Variable(LT(1,1).fill_(0))
    for i in range(0, 2000, args_bptt):
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, vocab_size)
#        _, idx = output_flat.data.max(1)
        y = output_flat.data.cpu().numpy()
        p = np.exp(y) / np.sum(np.exp(y))
        idx = np.random.choice(range(vocab_size), p=p.ravel())
        idx = LT([np.asscalar(idx)])
        predicted_idxs.append(idx)
        hidden = repackage_hidden(hidden)
        data = Variable(idx.view(1,1))
    return predicted_idxs
 
print("Vocab size is ({})\nLet's start.......".format(vocab_size))
predicted_idxs = sample_fn()
idxs = [predicted_idxs[i][0] for i in range(len(predicted_idxs))]
words = [corpus.dictionary.idx2word[idx] for idx in idxs]
print("Words summary ({}) ({}) ({})".format(len(words), type(words[0]), words[0]))
print("Words Begin ({}) Words End".format(words))
torch.set_printoptions(threshold=1000)
