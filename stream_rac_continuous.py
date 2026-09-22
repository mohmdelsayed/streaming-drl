import argparse
import math
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from obs_reward_transforms import NormalizeObservation, ScaleReward
from optimizer import StreamingOptimizer
from sparse_init import sparse_init

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, s):
        x = self.fc_layer(s)
        x = self.layernorm1(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = self.layernorm2(x)
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

    def distribution(self, s):
        mu, std = self.forward(s)
        return Independent(Normal(mu, std), 1)

    def sample_with_log_prob(self, s):
        dist = self.distribution(s)
        pre_action = dist.rsample()
        action = torch.tanh(pre_action)
        log_det = 2.0 * (math.log(2.0) - pre_action - F.softplus(-2.0 * pre_action))
        log_prob = dist.log_prob(pre_action) - log_det.sum(dim=-1)
        return action, log_prob

    def sample_action(self, s):
        action, _ = self.sample_with_log_prob(s)
        return action

class QFunction(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(QFunction, self).__init__()
        self.fc_layer = nn.Linear(n_obs + n_actions, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_layer = nn.Linear(hidden_size, 1)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.fc_layer(x)
        x = self.layernorm1(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = self.layernorm2(x)
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamRAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=256, lr_policy=1e-4, lr_value=3e-4, gamma=0.99, lamda=0.8, beta=0.99995, entropy_coeff=0.025, sparsity=0.95):
        super(StreamRAC, self).__init__()
        self.gamma = gamma
        self.alpha = entropy_coeff / n_actions
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.q_net = QFunction(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        sparse_init(self.modules(), target_sparsity=sparsity)
        self.optimizer_policy = StreamingOptimizer(self.policy_net.parameters(), lr=lr_policy, gamma=gamma, lamda=lamda, beta=beta)
        self.optimizer_q = StreamingOptimizer(self.q_net.parameters(), lr=lr_value, gamma=gamma, lamda=lamda, beta=beta)

    def q(self, s, a):
        return self.q_net(s, a)

    @torch.no_grad()
    def sample_action(self, s):
        s = torch.as_tensor(s, dtype=torch.float32)
        return self.policy_net.sample_action(s)

    def update_params(self, s, a, r, s_prime, terminated, truncated):
        reset = terminated or truncated
        done_mask = 0.0 if terminated else 1.0
        s = torch.tensor(np.array(s), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float32)
        done_mask = torch.tensor(np.array(done_mask), dtype=torch.float32)

        with torch.no_grad():
            a_target, log_prob_target = self.policy_net.sample_with_log_prob(s_prime)
            q_prime = self.q(s_prime, a_target)
            target_q = r + done_mask * self.gamma * (q_prime - self.alpha * log_prob_target)

        self.optimizer_policy.zero_grad()
        a_policy, log_prob_policy = self.policy_net.sample_with_log_prob(s)
        q_val = self.q(s, a_policy)
        policy_loss = (q_val - self.alpha * log_prob_policy)
        policy_loss.backward()
        self.optimizer_policy.step(reset=True)
        self.optimizer_policy.zero_grad()

        self.optimizer_q.zero_grad()
        q_val = self.q(s, a.detach())
        delta = target_q - q_val.detach()
        q_val.backward()
        self.optimizer_q.step(delta.item(), reset=reset)
        self.optimizer_q.zero_grad()
        return a_target

def main(env_name, seed, lr_policy, lr_value, gamma, lamda, beta, total_steps, entropy_coeff, hidden_size, debug, render=False, sparsity=0.95):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    reward_scaler = ScaleReward(gamma=gamma)
    observation_normalizer = NormalizeObservation(env.observation_space)
    agent = StreamRAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=hidden_size, lr_policy=lr_policy, lr_value=lr_value, gamma=gamma, lamda=lamda, beta=beta, entropy_coeff=entropy_coeff, sparsity=sparsity)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id), "entropy_coeff: {}".format(entropy_coeff))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    s = observation_normalizer.normalize(s)
    a_tensor = agent.sample_action(s)
    for t in range(1, total_steps + 1):
        s_prime, r, terminated, truncated, info = env.step(a_tensor.detach().cpu().numpy())
        s_prime = observation_normalizer.normalize(s_prime)
        r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
        a_prime = agent.update_params(s, a_tensor, r, s_prime, terminated, truncated)
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            s, _ = env.reset()
            s = observation_normalizer.normalize(s)
            a_tensor = agent.sample_action(s)
        else:
            s = s_prime; a_tensor = a_prime
    env.close()
    save_dir = "data_stream_rac_{}_lra{}_lrc{}_gamma{}_lamda{}_beta{}_entropy_coeff{}_hidden_size{}_sparsity{}".format(
        env.spec.id,
        lr_policy,
        lr_value,
        gamma,
        lamda,
        beta,
        entropy_coeff,
        hidden_size,
        sparsity,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StreamRAC')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_policy', type=float, default=1e-4, help='Learning rate for actor')
    parser.add_argument('--lr_value', type=float, default=3e-4, help='Learning rate for critic/Q network')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.025, help='Entropy coefficient')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr_policy, args.lr_value, args.gamma, args.lamda, args.beta, args.total_steps, args.entropy_coeff, args.hidden_size, args.debug, args.render, args.sparsity)