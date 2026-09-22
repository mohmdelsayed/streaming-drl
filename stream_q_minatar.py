import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from obs_reward_transforms import NormalizeObservation, ScaleReward
from optimizer import StreamingOptimizer
from sparse_init import sparse_init

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class StreamQ(nn.Module):
    def __init__(self, n_channels=4, n_actions=3, hidden_size=128, lr=3e-4, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.2, total_steps=5_000_000, gamma=0.99, lamda=0.8, beta=0.99995, sparsity=0.95):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        self.network = nn.Sequential(
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
        self.optimizer = StreamingOptimizer(self.parameters(), lr=lr, gamma=gamma, lamda=lamda, beta=beta)

    def q(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.moveaxis(x, -1, 0)
        return self.network(x)

    @torch.no_grad()
    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action, False
            return random_action, True
        q_values = self.q(s)
        return torch.argmax(q_values, dim=-1).item(), False

    def update_params(self, s, a, r, s_prime, terminated, truncated, is_nongreedy):
        reset = terminated or truncated or is_nongreedy
        done_mask = 0.0 if terminated else 1.0
        a = torch.as_tensor(a, dtype=torch.int64)
        r = torch.as_tensor(r, dtype=torch.float32)
        done_mask = torch.as_tensor(done_mask, dtype=torch.float32)

        q_sa = self.q(s)[a]
        with torch.no_grad():
            max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
            td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
            delta_value = float((td_target - q_sa).item())

        self.optimizer.zero_grad()
        q_sa.backward()
        self.optimizer.step(delta_value, reset=reset)

def main(env_name, seed, lr, gamma, lamda, beta, total_steps, epsilon_target, epsilon_start, exploration_fraction, hidden_size, debug, render=False, sparsity=0.95):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    reward_scaler = ScaleReward(gamma=gamma)
    observation_normalizer = NormalizeObservation(env.observation_space)
    agent = StreamQ(n_channels=env.observation_space.shape[-1], n_actions=env.action_space.n, hidden_size=hidden_size, lr=lr, gamma=gamma, lamda=lamda, beta=beta, epsilon_target=epsilon_target, epsilon_start=epsilon_start, exploration_fraction=exploration_fraction, total_steps=total_steps, sparsity=sparsity)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    episode_num = 1
    s, _ = env.reset(seed=seed)
    s = observation_normalizer.normalize(s)
    a, _ = agent.sample_action(s)
    for t in range(1, total_steps+1):
        s_prime, r, terminated, truncated, info = env.step(a)
        s_prime = observation_normalizer.normalize(s_prime)
        r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
        if terminated or truncated:
            a_prime, next_is_nongreedy = None, False
        else:
            a_prime, next_is_nongreedy = agent.sample_action(s_prime)
        agent.update_params(s, a, r, s_prime, terminated, truncated, next_is_nongreedy)
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}, Episode Number {}, Epsilon {}".format(info['episode']['r'][0], t, episode_num, agent.epsilon))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            s, _ = env.reset()
            s = observation_normalizer.normalize(s)
            a, _ = agent.sample_action(s)
            episode_num += 1
        else:
            s = s_prime; a = a_prime
    env.close()
    save_dir = "data_stream_q_{}_lr{}_gamma{}_lamda{}_beta{}_hidden_size{}_sparsity{}".format(env.spec.id, lr, gamma, lamda, beta, hidden_size, sparsity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream Q(λ)')
    parser.add_argument('--env_name', type=str, default='MinAtar/Breakout-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.2)
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.beta, args.total_steps, args.epsilon_target, args.epsilon_start, args.exploration_fraction, args.hidden_size, args.debug, args.render, args.sparsity)