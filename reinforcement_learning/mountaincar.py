# -*- coding: utf-8 -*-
import argparse
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.envs.registration import register


class BaseAgent:
    _model_ready = False
    _model = None

    def __init__(self, state_size, action_size, max_memory_length=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=max_memory_length)
        self.gamma = 0.9
        self.epsilon = 0.99
        self.e_decay = .99
        self.e_min = 0.05

    def init_model(self):
        raise NotImplementedError()

    def _fit_model(self, x, y, batch_size):
        raise NotImplementedError()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon or not self._model_ready:
            return random.randrange(self.action_size)

        action_values = self.predict_action_values(state).detach().numpy()
        return np.argmax(action_values)

    def predict_action_values(self, state):
        _state = self.get_usable_state(state)
        action_values = self._model(_state)
        return action_values

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        idx_batch = set(random.sample(range(len(self.memory)), batch_size))
        mini_batch = [val for i, val in enumerate(self.memory) if i in idx_batch]
        x = []
        y = []
        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.predict_action_values(state)
            if done:
                target[action] = reward
            else:
                next_action_values = self.predict_action_values(next_state)
                next_action_value = torch.max(next_action_values).item()
                next_action_value = reward + self.gamma * next_action_value
                target[action] = next_action_value
            x.append(state)
            y.append(target)
        self._fit_model(x, y, batch_size)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, path):
        self._model = torch.load(path)

    def save(self, path):
        torch.save(self._model, path)

    def get_usable_state(self, state):
        raise NotImplementedError()


class DQNAgentModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=28):
        super().__init__()
        self.state_layer = nn.Linear(state_size, hidden_units)
        self.hidden_layer = nn.Linear(hidden_units, hidden_units)
        self.action_layer = nn.Linear(hidden_units, action_size)

    def forward(self, x):
        x = F.tanh(self.state_layer(x))
        x = F.tanh(self.hidden_layer(x))
        action_scores = self.action_layer(x)
        return action_scores


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, neurons, max_memory_length=100000):
        super().__init__(state_size, action_size, max_memory_length)
        self.dense_neurons = neurons
        self.train_epochs = 5

    def init_model(self):
        self._model_ready = True
        self._model = DQNAgentModel(self.state_size, self.action_size, self.dense_neurons)
        self.optimizer = optim.Adam(self._model.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self._model.eval()

    def get_usable_state(self, state):
        return torch.from_numpy(state[-1]).float()

    def _fit_model(self, x, y, batch_size):
        losses = []
        self._model.train()
        for epoch in range(self.train_epochs):
            for state, target in zip(x, y):
                self.optimizer.zero_grad()
                _state = self.get_usable_state(state)
                _target = target.detach()
                out = self._model(_state)
                loss = nn.MSELoss()(out, _target)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
        self._model.eval()


class LSTMAgentModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.lstm1 = nn.LSTMCell(state_size, hidden_units)
        self.lstm2 = nn.LSTMCell(hidden_units, hidden_units)
        self.linear = nn.Linear(hidden_units, action_size)

    def forward(self, x):
        h_t = torch.zeros(1, self.hidden_units, dtype=torch.float)
        c_t = torch.zeros(1, self.hidden_units, dtype=torch.float)
        h_t2 = torch.zeros(1, self.hidden_units, dtype=torch.float)
        c_t2 = torch.zeros(1, self.hidden_units, dtype=torch.float)
        for i, input_t in enumerate(x.chunk(x.size(0), dim=0)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        action_scores = self.linear(h_t2)
        return action_scores[-1]


class LSTMAgent(DQNAgent):
    def init_model(self):
        self._model_ready = True
        self._model = LSTMAgentModel(self.state_size, self.action_size, self.dense_neurons)
        self.optimizer = optim.Adam(self._model.parameters(), lr=3e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self._model.eval()

    def get_usable_state(self, state):
        return torch.from_numpy(state).float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=int, default=256, help='replay size')
    parser.add_argument('--trials', type=int, default=5000, help='number of trials')
    parser.add_argument('--lstm', action='store_true', help='use LSTM based agent')
    parser.add_argument('--negative', action='store_true', help='mark as negative results with reward significantly lower than maximum')
    parser.add_argument('--neurons', type=int, default=24, help='nn layer size')
    opts = parser.parse_args()

    register(
        id='MountainCarLong-v0',
        entry_point='gym.envs.classic_control.mountain_car:MountainCarEnv',
        trials=opts.trials,
        max_episode_steps=200
    )
    env = gym.make('MountainCarLong-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    if opts.lstm:
        agent = LSTMAgent(state_size, action_size, neurons=opts.neurons)
    else:
        agent = DQNAgent(state_size, action_size, neurons=opts.neurons)

    agent.e_min = 0.01
    agent.init_model()
    max_score = -1
    for e in range(opts.trials):
        state = np.reshape(env.reset(), [1, state_size])
        to_remember = []
        max_reward = 0
        states = None
        env_steps = 2000
        for time in range(env_steps):
            env.render()
            if states is None:
                states = np.vstack((state))
            else:
                states = np.vstack((states, state))
            action = agent.act(states)
            next_state, reward, done, info = env.step(action)
            reward = reward * 10 if reward >= 0 else (1 + next_state[0])
            max_reward = max(max_reward, reward)
            next_state = np.reshape(next_state, [1, state_size])
            to_remember.append([states, action, reward, next_state, done])
            state = next_state
            max_score = max(max_score, reward)
            if done:
                break
        if opts.negative:
            if max_reward < (max_score / 1.5):
                print("negative sample")
                for rec in to_remember:
                    rec[2] = -10
        for rec in to_remember:
            agent.remember(rec[0], rec[1], rec[2], rec[3], rec[4])
        print("- trial: {}/{}, score: {:.2f} (max: {:.2f}), e: {:.2}"
              .format(e, opts.trials, max_reward, max_score, agent.epsilon))
        agent.replay(opts.replay)
