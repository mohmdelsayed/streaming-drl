import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from wrappers import NormalizeObservation, ScaleReward, AddTimeInfo
from sparse_init import sparse_init
from cnn_policies import SSEncoderModel
from pixel_ant import VisualAntReacher


class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, encoder, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.fc_layer   = nn.Linear(encoder.latent_dim, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, img, prop):
        x = self.encoder(img, prop, random_rad=True, detach=True)
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, encoder, hidden_size=128):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.fc_layer   = nn.Linear(encoder.latent_dim, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, img, prop):
        x = self.encoder(img, prop, random_rad=True, detach=False)
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, encoder, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.encoder = encoder
        self.policy_net = Actor(encoder=encoder, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(encoder=encoder, hidden_size=hidden_size)
        self.optimizer_policy = ObGD(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = ObGD(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, img, prop):
        return self.policy_net(img, prop)

    def v(self, img, prop):
        return self.value_net(img, prop)

    def sample_action(self, s):
        img, prop = s.image, s.proprioception
        img = torch.from_numpy(img).float().unsqueeze(0)
        prop = torch.from_numpy(prop).float().unsqueeze(0)
        mu, std = self.pi(img, prop)
        dist = Normal(mu, std)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        a, r, done_mask = torch.tensor(np.array(a)), torch.tensor(np.array(r)), \
                                torch.tensor(np.array(done_mask), dtype=torch.float)
        
        img, prop = s.image, s.proprioception
        img = torch.from_numpy(img).float().unsqueeze(0)
        prop = torch.from_numpy(prop).float().unsqueeze(0)
        next_img, next_prop = s.image, s_prime.proprioception
        next_img = torch.from_numpy(next_img).float().unsqueeze(0)
        next_prop = torch.from_numpy(next_prop).float().unsqueeze(0)

        v_s, v_prime = self.v(img, prop), self.v(next_img, next_prop)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        mu, std = self.pi(img, prop)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, render=False):
    torch.manual_seed(seed); np.random.seed(seed)
    env = VisualAntReacher()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
     
    encoder = SSEncoderModel(env.image_space.shape, [env.proprioception_space.shape[0]+1], args.net_params, args.rad_offset, spatial_softmax=args.no_spatial_softmax)
    agent = StreamAC(encoder=encoder, n_actions=env.action_space.shape[0], lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env_name))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime,  terminated or truncated, entropy_coeff, overshooting_info)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {:.3f}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
    env.close()
    save_dir = "data_stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--rad_offset', type=float, default=0.02)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--no_spatial_softmax', action='store_false')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # TODO: Remove hard-coding
    args.net_params = {
        'conv': [
            # in_channel, out_channel, kernel_size, stride
            [-1, 32, 3, 2],
            [32, 64, 3, 2],
            [64, 64, 3, 2],
        ],

        'latent': 50,   # Only applicable if spatial softmax isn't used
    }
    args.env_name = "VisualAntReacher"

    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.entropy_coeff, args.kappa_policy, args.kappa_value, args.debug, args.overshooting_info, args.render)