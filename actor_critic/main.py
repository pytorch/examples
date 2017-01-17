import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=int, default=0.999, metavar='G',
                    help='discount factor (default: 0.999)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['action', 'value'])
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 16)
        self.affine2 = nn.Linear(16, 32)
        self.action_head = nn.Linear(32, 2)
        self.value_head = nn.Linear(32, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


model = Policy()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    action = probs.multinomial()
    model.saved_actions.append(SavedAction(action, state_value))
    return action.data


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    value_loss = 0
    for (action, value), r in zip(saved_actions[::-1], model.rewards[::-1]):
        R = r + args.gamma * R
        action.reinforce(R - value.data.squeeze())
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([R])))
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    gradients = [torch.ones(1)] + [None] * len(saved_actions)
    autograd.backward(final_nodes, gradients)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


episode_durations = []
for i_episode in count(1):
    state = env.reset()
    for t in count(1):
        action = select_action(state)
        state, reward, done, _ = env.step(action[0,0])
        if args.render:
            env.render()
        model.rewards.append(reward)
        if done:
            break

    finish_episode()
    episode_durations.append(t)
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, torch.Tensor(episode_durations[-100:]).mean()))
