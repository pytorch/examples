import argparse
import random
from collections import deque

import gym

import numpy as np

import torch as T
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


class LinearOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_structure, output_dim):
        super(LinearOutputMLP, self).__init__()
        all_layers_dim = [input_dim] + hidden_structure
        layers = []

        for idx in range(len(all_layers_dim) - 1):
            n_in = all_layers_dim[idx]
            n_out = all_layers_dim[idx + 1]

            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.PReLU())
            # layers.append(nn.BatchNorm1d(n_out))
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(n_out, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class QNetwork(object):
    def __init__(
            self,
            input_dim,
            n_actions,
            replay_len=1000,
            gamma=0.9,
            epsilon=0.1,
            epsilon_decay=0.99,
            n_samples=100,
    ):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.n_samples = n_samples

        # the Q network
        self.model = LinearOutputMLP(input_dim, [128, 128], n_actions)
        self.model_eval = LinearOutputMLP(input_dim, [128, 128], n_actions)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

        self.experience = deque(maxlen=replay_len)

    def get_action(self, state):
        pred = self.model(state)
        if np.random.random() < self.epsilon:
            self.epsilon *= self.epsilon_decay
            return T.from_numpy(np.random.randint(self.n_actions, size=(1, 1)))
        return pred.data.max(1)[1].cpu()

    def update(self):
        sampled_experience = []
        for _ in range(self.n_samples):
            sampled_experience.append(random.choice(self.experience))

        self.model_eval.load_state_dict(self.model.state_dict())
        loss = Variable(
            T.FloatTensor([0.]),
            requires_grad=True,
            # volatile=False,
        )
        for state, action, reward, next_state, done in sampled_experience:
            reward = Variable(T.FloatTensor([reward]))
            if done:
                next_q = reward
            else:
                next_q = self.gamma * self.model_eval(next_state).max() + reward
            current_q = current_q = self.model(state)[0][action[0][0]]
            loss = loss + (next_q - current_q) ** 2

        loss /= self.n_samples
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def record(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))


def main():

    parser = argparse.ArgumentParser(description='PyTorch DQN example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--epsilon', type=float, default=0.5, metavar='G',
                        help='proportion of randomly selected actions (default: 0.1)')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, metavar='G',
                        help='decay of epsilon (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    T.manual_seed(args.seed)

    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]

    q_value = QNetwork(
        input_dim,
        n_actions,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        replay_len=50000,
        n_samples=64,
    )

    running_reward = 0
    for i_episode in range(1000):
        state = env.reset()
        for time in range(10000):
            if isinstance(state, np.ndarray):
                state = Variable(T.from_numpy(state).float().unsqueeze(0))
            action = q_value.get_action(state)
            new_state, reward, done, _ = env.step(action[0][0])
            if args.render:
                env.render()

            x, x_dot, theta, theta_dot = new_state

            # Reward modification reference:
            # https://morvanzhou.github.io/tutorials/machine-learning/torch/4-05-DQN/
            r1 = (env.env.x_threshold - abs(x)) / env.env.x_threshold - 0.8
            r2 = (env.env.theta_threshold_radians - abs(theta)) / env.env.theta_threshold_radians - 0.5
            reward = r1 + r2

            new_state = Variable(T.from_numpy(new_state).float().unsqueeze(0))
            q_value.record(state, action, reward, new_state, done)
            state = new_state

            # use the first 20 episode to collect data
            if i_episode > 100:
                q_value.update()

            if done:
                break

        # logging
        running_reward += time
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode,
                time,
                running_reward / args.log_interval,
            ))
            running_reward = 0
            if running_reward > 200:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, time))
                break


if __name__ == '__main__':
    main()
