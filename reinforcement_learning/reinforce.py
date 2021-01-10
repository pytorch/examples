import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--env', type=str, default="CartPole-v0", metavar='E',
                    help='environment (default: CartPole-v0)')

parser.add_argument('--discount_factor', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--hidden_size', type=int, default=128, metavar='H',
                    help='number of hidden units for the policy network\'s input layer (default: 128)')

parser.add_argument('--learning_rate', type=float, default=1e-2, metavar='L',
                    help='learning rate for the Adam optimizer (default: 1e-2)')

parser.add_argument('--dropout', type=float, default=0.6, metavar='D',
                    help='Dropout probability for the policy network (default: 0.6)')

parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')

parser.add_argument('--render', action='store_true',
                    help='render the environment')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()


class PolicyNetwork(nn.Module):

    """
        Implements the policy for the REINFORCE algorithm
    """

    def __init__(self, num_features, num_actions, hidden_size, learning_rate, dropout_prob):
        super(PolicyNetwork, self).__init__()
        self._input_layer = nn.Linear(num_features, hidden_size)
        self._dropout = nn.Dropout(p=dropout_prob)
        self._output_layer = nn.Linear(hidden_size, num_actions)
        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X):
        X = self._input_layer(X)
        X = self._dropout(X)
        X = F.relu(X)
        action_logits = self._output_layer(X)
        policy = F.softmax(action_logits, dim=1)
        return policy

    def update(self, returns, action_logits):
        self._optimizer.zero_grad()
        loss = self.loss_fn(returns, action_logits)
        loss.backward()
        self._optimizer.step()

    @staticmethod
    def loss_fn(returns, action_logits):
        batch_size = len(returns)
        policy_losses = [-action_logits[b] * returns[b] for b in range(batch_size)]
        loss = torch.cat(policy_losses).sum() / batch_size
        return loss

class ActorCriticAgent:

    """
        Implements the concept of agent
        (action/reinforcement interface + internal state)
    """

    def __init__(self, num_features, num_actions, hidden_size, learning_rate, dropout_prob, discount_factor):
        self._network = PolicyNetwork(num_features, num_actions, hidden_size, learning_rate, dropout_prob)
        self._gamma = discount_factor
        self._rewards_buffer = []
        self._action_logits_buffer = []

    def action(self, state):

        x = torch.from_numpy(state).float().unsqueeze(0)
        policy = self._network(x)

        policy = Categorical(policy)
        action = policy.sample()

        self._action_logits_buffer.append(policy.log_prob(action))

        return action.item()

    def reinforce(self, reward, terminal):
        self._rewards_buffer.append(reward)
        if terminal:
            returns = self.compute_returns()
            self._network.update(returns, self._action_logits_buffer)
            self._rewards_buffer.clear()
            self._action_logits_buffer.clear()

    def compute_returns(self, nan_preventing_eps=np.finfo(np.float32).eps.item()):
        returns = []
        current_return = 0
        for reward in reversed(self._rewards_buffer):
            current_return = reward + self._gamma * current_return
            returns = [current_return] + returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + nan_preventing_eps)
        return returns

def run_episode(agent, env, render):

    """
        Runs a full episode on the environment
    """

    ep_reward = 0
    ep_steps = 0

    state = env.reset()
    terminal = False
    while not terminal:

        action = agent.action(state)
        state, reward, terminal, _ = env.step(action)

        agent.reinforce(reward, terminal)

        ep_reward += reward
        ep_steps += 1

        if render:
            env.render()

    return ep_reward, ep_steps

if __name__ == '__main__':

    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = ActorCriticAgent(
        num_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout,
        discount_factor=args.discount_factor
    )

    running_reward = 10

    # Run infinitely many episodes
    for episode in count(1):

        ep_reward, ep_steps = run_episode(agent, env, args.render)

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Log results
        if episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                episode, ep_reward, running_reward))

        # Check if we have "solved" the problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, ep_steps))
            break
