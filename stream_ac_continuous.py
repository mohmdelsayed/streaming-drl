import argparse
import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optimizer import StreamingOptimizer
from obs_reward_transforms import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.layernorm1(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = self.layernorm2(x)
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear_layer  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.layernorm1(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = self.layernorm2(x)
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr_policy=1e-4, lr_value=2e-4, gamma=0.99, lamda=0.8, beta=0.99995, sparsity=0.95):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        sparse_init(self.modules(), target_sparsity=sparsity)
        self.optimizer_policy = StreamingOptimizer(self.policy_net.parameters(), lr=lr_policy, gamma=gamma, lamda=lamda, beta=beta)
        self.optimizer_value = StreamingOptimizer(self.value_net.parameters(), lr=lr_value, gamma=gamma, lamda=lamda, beta=beta)

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    @torch.no_grad()
    def sample_action(self, s):
        x = torch.as_tensor(s, dtype=torch.float32)
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, terminated, truncated, entropy_coeff):
        reset = terminated or truncated
        done_mask = 0.0 if terminated else 1.0
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        s_prime = torch.as_tensor(s_prime, dtype=torch.float32)
        done_mask = torch.as_tensor(done_mask, dtype=torch.float32)

        v_s = self.v(s)
        with torch.no_grad():
            v_prime = self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta = td_target - v_s
            delta_value = float(delta.item())
            delta_sign = 1.0 if delta_value > 0.0 else -1.0 if delta_value < 0.0 else 0.0

        mu, std = self.pi(s)
        dist = Normal(mu, std)

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
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    reward_scaler = ScaleReward(gamma=gamma)
    observation_normalizer = NormalizeObservation(env.observation_space)
    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=hidden_size, lr_policy=lr_policy, lr_value=lr_value, gamma=gamma, lamda=lamda, beta=beta, sparsity=sparsity)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    s = observation_normalizer.normalize(s)
    for step in range(1, total_steps + 1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        s_prime = observation_normalizer.normalize(s_prime)
        r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
        agent.update_params(s, a, r, s_prime, terminated, truncated, entropy_coeff)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], step))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(step)
            s, _ = env.reset()
            s = observation_normalizer.normalize(s)
    env.close()
    save_dir = "data_stream_ac_{}_lr_policy{}_lr_value{}_gamma{}_lamda{}_beta{}_entropy_coeff{}_hidden_size{}_sparsity{}".format(env.spec.id, lr_policy, lr_value, gamma, lamda, beta, entropy_coeff, hidden_size, sparsity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_policy', type=float, default=1e-4)
    parser.add_argument('--lr_value', type=float, default=2e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr_policy, args.lr_value, args.gamma, args.lamda, args.beta, args.total_steps, args.entropy_coeff, args.hidden_size, args.debug, args.render, args.sparsity)