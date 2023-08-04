import argparse
import gym
import numpy as np
import os
from itertools import chain, count

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical

TOTAL_EPISODE_STEP = 5000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

parser = argparse.ArgumentParser(description='PyTorch RPC RL example')
parser.add_argument('--world-size', type=int, default=2, metavar='W',
                    help='world size for RPC, rank 0 is the agent, others are observers')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _async_remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

class ObserverPolicy(nn.Module):
    def __init__(self):
        super(ObserverPolicy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        return self.affine2(x)


class AgentPolicy(nn.Module):
    def __init__(self):
        super(AgentPolicy, self).__init__()

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, action_scores):
        return F.softmax(action_scores, dim=1)


class Observer:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)
        self.policy = ObserverPolicy()

    def get_gradients(self, ctx_id):
        all_grads = dist_autograd.get_gradients(ctx_id)
        out_grads = []
        for p in self.policy.parameters():
            out_grads.append(all_grads[p])
        return out_grads

    def update_model(self, params):
        for p, new_p in zip(self.policy.parameters(), params):
            with torch.no_grad():
                p.set_(new_p)


    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Arguments:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        state, ep_reward = self.env.reset(), 0
        for step in range(n_steps):
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_scores = self.policy(state)
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, action_scores)

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            if done:
                break

class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.ob_policy = ObserverPolicy()
        self.policy = AgentPolicy()
        params = chain(self.ob_policy.parameters(), self.policy.parameters())
        self.optimizer = optim.Adam(params, lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, action_scores):
        probs = self.policy(action_scores)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each oberser to run n_steps.
        """
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def finish_episode(self, ctx_id):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        dist_autograd.backward([policy_loss])

        # collect gradients from observers
        futs = []
        # should be able to optimize this loop with a gather
        for ob_rref in self.ob_rrefs:
            futs.append(_async_remote_method(Observer.get_gradients, ob_rref, ctx_id))

        grads = [fut.wait() for fut in futs]
        grads = [*zip(*grads)]
        grads = [sum(grad) for grad in grads]

        # we can add a mode to directly accumulate gradients on the model
        # instead of putting it into a separate ctx
        # set grads for observer model
        ob_params = list(self.ob_policy.parameters())
        for g, p in zip(grads, ob_params):
            p.grad = g

        # set grads for agent model
        ctx_grads = dist_autograd.get_gradients(ctx_id)
        for p in self.policy.parameters():
            p.grad = ctx_grads[p]

        self.optimizer.step()

        ob_params = [p.data.detach() for p in self.ob_policy.parameters()]
        # should be able to optimize this loop with a scatter
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(_async_remote_method(Observer.update_model, ob_rref, ob_params))
        for fut in futs:
            fut.wait()


        return min_reward


def run_worker(rank, world_size):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size)
        for i_episode in count(1):
            n_steps = int(TOTAL_EPISODE_STEP / (args.world_size - 1))
            with dist_autograd.context() as ctx_id:
                agent.run_episode(n_steps=n_steps)
                last_reward = agent.finish_episode(ctx_id)

                if i_episode % args.log_interval == 0:
                    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                          i_episode, last_reward, agent.running_reward))

                if agent.running_reward > agent.reward_threshold:
                    print("Solved! Running reward is now {}!".format(agent.running_reward))
                    break
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from agents
    rpc.shutdown()


def main():
    mp.spawn(
        run_worker,
        args=(args.world_size, ),
        nprocs=args.world_size,
        join=True
    )

if __name__ == '__main__':
    main()
