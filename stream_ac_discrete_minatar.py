import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from obs_reward_transforms import NormalizeObservation, ScaleReward
from optimizer import StreamingOptimizer
from sparse_init import sparse_init

class StreamAC(nn.Module):
    def __init__(self, n_channels=4, n_actions=3, hidden_size=128, lr_policy=1e-4, lr_value=3e-4, gamma=0.99, lamda=0.8, beta=0.99995, sparsity=0.95):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.value_net = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=1),
            nn.Flatten(start_dim=0),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Linear(1024, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.policy_net = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=1),
            nn.Flatten(start_dim=0),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Linear(1024, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        sparse_init(self.modules(), target_sparsity=sparsity)
        self.optimizer_policy = StreamingOptimizer(self.policy_net.parameters(), lr=lr_policy, gamma=gamma, lamda=lamda, beta=beta)
        self.optimizer_value = StreamingOptimizer(self.value_net.parameters(), lr=lr_value, gamma=gamma, lamda=lamda, beta=beta)

    def pi(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.moveaxis(x, -1, 0)
        preferences = self.policy_net(x)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.moveaxis(x, -1, 0)
        return self.value_net(x)

    @torch.no_grad()
    def sample_action(self, s):
        probs = self.pi(s)
        dist = Categorical(probs)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, terminated, truncated, entropy_coeff):
        reset = terminated or truncated
        done_mask = 0.0 if terminated else 1.0
        a = torch.as_tensor(a, dtype=torch.int64)
        r = torch.as_tensor(r, dtype=torch.float32)
        done_mask = torch.as_tensor(done_mask, dtype=torch.float32)

        v_s = self.v(s)
        with torch.no_grad():
            v_prime = self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta = td_target - v_s
            delta_value = float(delta.item())
            delta_sign = 1.0 if delta_value > 0.0 else -1.0 if delta_value < 0.0 else 0.0

        probs = self.pi(s)
        dist = Categorical(probs)

        log_prob_pi = dist.log_prob(a).sum()
        entropy_pi = entropy_coeff * dist.entropy().sum() * delta_sign
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        v_s.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta_value, reset=reset)
        self.optimizer_value.step(delta_value, reset=reset)

def main(env_name, seed, lr_policy, lr_value, gamma, lamda, beta, total_steps, entropy_coeff, hidden_size, debug, render=False, sparsity=0.95):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    reward_scaler = ScaleReward(gamma=gamma)
    observation_normalizer = NormalizeObservation(env.observation_space)
    agent = StreamAC(n_channels=env.observation_space.shape[-1], n_actions=env.action_space.n, hidden_size=hidden_size, lr_policy=lr_policy, lr_value=lr_value, gamma=gamma, lamda=lamda, beta=beta, sparsity=sparsity)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    s = observation_normalizer.normalize(s)
    episode_num = 1
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        s_prime = observation_normalizer.normalize(s_prime)
        r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
        agent.update_params(s, a, r, s_prime, terminated, truncated, entropy_coeff)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}, Episode Number {}".format(info['episode']['r'][0], t, episode_num))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            s, _ = env.reset()
            s = observation_normalizer.normalize(s)
            episode_num += 1
    env.close()
    save_dir = "data_stream_ac_{}_lr_policy{}_lr_value{}_gamma{}_lamda{}_beta{}_entropy_coeff{}_hidden_size{}_sparsity{}".format(env.spec.id, lr_policy, lr_value, gamma, lamda, beta, entropy_coeff, hidden_size, sparsity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='MinAtar/Breakout-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_policy', type=float, default=1e-4)
    parser.add_argument('--lr_value', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr_policy, args.lr_value, args.gamma, args.lamda, args.beta, args.total_steps, args.entropy_coeff, args.hidden_size, args.debug, args.render, args.sparsity)