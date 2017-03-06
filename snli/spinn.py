import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import itertools


def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two or three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None):
        super(Reduce, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided as
        iterables and batched internally into tensors.

        Additionally augments each new node with pointers to its children.

        Args:
            left_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~autograd.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left`` and ``right``
                attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])
        out = unbundle(tree_lstm(left[1], right[1], lstm_in))
        # for o, l, r in zip(out, left_in, right_in):
        #     o.left, o.right = l, r
        return out


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, predict):
        super(Tracker, self).__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        if predict:
            self.transition = nn.Linear(tracker_size, 4)
        self.state_size = tracker_size

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [Variable(
                x.data.new(x.size(0), self.state_size).zero_())] 
        self.state = self.rnn(x, self.state)
        if hasattr(self, 'transition'):
            return unbundle(self.state), self.transition(self.state[0])
        return unbundle(self.state), None

class SPINN(nn.Module):

    def __init__(self, config):
        super(SPINN, self).__init__()
        self.config = config
        assert config.d_hidden == config.d_proj / 2
        self.reduce = Reduce(config.d_hidden, config.d_tracker)
        if config.d_tracker is not None:
            self.tracker = Tracker(config.d_hidden, config.d_tracker,
                                   predict=config.predict)

    def forward(self, buffers, transitions):
        buffers = [list(torch.split(b.squeeze(1), 1, 0))
                   for b in torch.split(buffers, 1, 1)]
        stacks = [[buf[0], buf[0]] for buf in buffers]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        else:
            assert transitions is not None

        if transitions is not None:
            num_transitions = transitions.size(0)
            # trans_loss, trans_acc = 0, 0
        else:
            num_transitions = len(buffers[0]) * 2 - 3

        for i in range(num_transitions):
            if transitions is not None:
                trans = transitions[i]
            if hasattr(self, 'tracker'):
                tracker_states, trans_hyp = self.tracker(buffers, stacks)
                if trans_hyp is not None:
                    trans = trans_hyp.max(1)[1]
                    # if transitions is not None:
                    #     trans_loss += F.cross_entropy(trans_hyp, trans)
                    #     trans_acc += (trans_preds.data == trans.data).mean()
                    # else:
                    #     trans = trans_preds
            else:
                tracker_states = itertools.repeat(None)
            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)
            for transition, buf, stack, tracking in batch:
                if transition == 3:  # shift
                    stack.append(buf.pop())
                elif transition == 2:  # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)
            if rights:
                reduced = iter(self.reduce(lefts, rights, trackings))
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        stack.append(next(reduced))
        # if trans_loss is not 0:
        return bundle([stack.pop() for stack in stacks])[0]
