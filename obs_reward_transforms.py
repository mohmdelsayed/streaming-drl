import gymnasium as gym
import numpy as np

class SampleMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float32")
        self.var = np.ones(shape, "float32")
        self.p = np.ones(shape, "float32")
        self.count = 0

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = np.zeros_like(x)
        self.mean, self.var, self.p, self.count = self.update_mean_var_count_from_moments(self.mean, self.p, self.count, x*1.0)

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count

class NormalizeObservation:
    def __init__(self, observation_space: gym.Space, epsilon: float = 1e-8, transform=None, shared_axis=None):
        self.shared_axis = shared_axis
        if shared_axis is None:
            stats_shape = observation_space.shape
        else:
            # for frame-stacked observations, we want to compute the mean and variance across individual frames
            stats_shape = tuple(d for i, d in enumerate(observation_space.shape) if i != shared_axis)
        self.obs_stats = SampleMeanStd(shape=stats_shape)
        self.transform = transform
        self.epsilon = epsilon

    def normalize(self, obs):
        if self.transform is not None:
            obs = self.transform(obs)
        obs = np.asarray(obs, dtype=np.float32)
        if self.shared_axis is None:
            self.obs_stats.update(obs)
            return (obs - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + self.epsilon)
        for frame in np.moveaxis(obs, self.shared_axis, 0):
            # for frame-stacked observations, update the stats for each frame separately
            self.obs_stats.update(frame)
        mean = np.expand_dims(self.obs_stats.mean, self.shared_axis)
        var = np.expand_dims(self.obs_stats.var, self.shared_axis)
        return (obs - mean) / np.sqrt(var + self.epsilon)

class ScaleReward:
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.reward_stats = SampleMeanStd(shape=())
        self.reward_trace = 0.0
        self.gamma = gamma
        self.epsilon = epsilon

    def scale(self, reward, terminated=False, truncated=False):
        self.reward_trace = self.reward_trace * self.gamma + reward
        self.reward_stats.update(np.asarray(self.reward_trace, dtype=np.float32))
        scaled_reward = reward / np.sqrt(self.reward_stats.var + self.epsilon)
        if terminated or truncated:
            self.reward_trace = 0.0
        return scaled_reward
