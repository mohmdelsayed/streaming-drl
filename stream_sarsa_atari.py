import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from obs_reward_transforms import NormalizeObservation, ScaleReward
from optimizer import StreamingOptimizer
from sparse_init import sparse_init

torch.set_default_dtype(torch.float64)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class StreamSARSA(nn.Module):
    def __init__(self, n_actions=3, hidden_size=256, lr=3e-4, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.05, total_steps=50_000_000, gamma=0.99, lamda=0.8, beta=0.99995, sparsity=0.95):
        super(StreamSARSA, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=5),
            nn.LayerNorm([32, 16, 16], elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=3),
            nn.LayerNorm([64, 5, 5], elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.LayerNorm([64, 2, 2], elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        sparse_init(self.modules(), target_sparsity=sparsity)
        self.optimizer = StreamingOptimizer(self.parameters(), lr=lr, gamma=gamma, lamda=lamda, beta=beta)

    def q(self, x):
        x = torch.as_tensor(np.array(x), dtype=torch.float64)
        return self.network(x)

    @torch.no_grad()
    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        q_values = self.q(s)
        return torch.argmax(q_values, dim=-1).item()

    def update_params(self, s, a, r, s_prime, a_prime, terminated, truncated):
        reset = terminated or truncated
        done_mask = 0.0 if terminated else 1.0
        a = torch.as_tensor(a, dtype=torch.int64)
        r = torch.as_tensor(r, dtype=torch.float64)
        a_prime = torch.as_tensor(a_prime, dtype=torch.int64)
        done_mask = torch.as_tensor(done_mask, dtype=torch.float64)
        q_sa = self.q(s)[a]
        with torch.no_grad():
            q_s_prime_a_prime = self.q(s_prime)[a_prime]
            td_target = r + self.gamma * q_s_prime_a_prime * done_mask
            delta_value = float((td_target - q_sa).item())
        self.optimizer.zero_grad()
        q_sa.backward()
        self.optimizer.step(delta_value, reset=reset)

def main(env_name, seed, lr, gamma, lamda, beta, total_steps, epsilon_target, epsilon_start, exploration_fraction, hidden_size, debug, render=False, sparsity=0.95):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    reward_scaler = ScaleReward(gamma=gamma)
    observation_normalizer = NormalizeObservation(env.observation_space, shared_axis=0)
    # observation_normalizer = NormalizeObservation(env.observation_space, transform=lambda obs: np.asarray(obs, dtype=np.float32) / 255.0)
    agent = StreamSARSA(n_actions=env.action_space.n, hidden_size=hidden_size, lr=lr, gamma=gamma, lamda=lamda, beta=beta, epsilon_target=epsilon_target, epsilon_start=epsilon_start, exploration_fraction=exploration_fraction, total_steps=total_steps, sparsity=sparsity)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    s = observation_normalizer.normalize(s)
    episode_num = 1
    a = agent.sample_action(s)
    for t in range(1, total_steps+1):
        s_prime, r, terminated, truncated, info = env.step(a)
        s_prime = observation_normalizer.normalize(s_prime)
        r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
        if terminated:
            a_prime = 0
        else:
            a_prime = agent.sample_action(s_prime)
        agent.update_params(s, a, r, s_prime, a_prime, terminated, truncated)
        if info and "episode" in info:
            if debug:
                print("Episodic Return: {}, Time Step {}, Episode Number {}, Epsilon {}".format(info['episode']['r'][0], t, episode_num, agent.epsilon))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            episode_num += 1
        if terminated or truncated:
            s, _ = env.reset()
            s = observation_normalizer.normalize(s)
            a = agent.sample_action(s)
        else:
            s = s_prime; a = a_prime
    env.close()
    save_dir = "data_stream_sarsa_{}_lr{}_gamma{}_lamda{}_beta{}_hidden_size{}_sparsity{}".format(env.spec.id, lr, gamma, lamda, beta, hidden_size, sparsity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream SARSA(λ)')
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.05)
    parser.add_argument('--total_steps', type=int, default=50_000_000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.beta, args.total_steps, args.epsilon_target, args.epsilon_start, args.exploration_fraction, args.hidden_size, args.debug, args.render, args.sparsity)