import numpy as np
import gymnasium as gym

class SampleMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.p = np.ones(shape, "float64")
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

class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.obs_stats = SampleMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs.proprioception = self.normalize(np.array([obs.proprioception]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs.proprioception = self.normalize(np.array([obs.proprioception]))[0]
        return obs, info

    def normalize(self, obs):
        self.obs_stats.update(obs)
        return (obs - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + self.epsilon)

class ScaleReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False
        self.reward_stats = SampleMeanStd(shape=())
        self.reward_trace = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        term = terminateds or truncateds
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        self.reward_stats.update(self.reward_trace)
        return rews / np.sqrt(self.reward_stats.var + self.epsilon)
    

class AddTimeInfo(gym.core.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if self.env.num_envs > 1:
            raise ValueError("AddTimeInfo only supports single environments")
        self.epi_time = -0.5
        self.time_limit = 1000
        
        self.obs_space_size = self.observation_space.shape[0] + self.env.num_envs
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,), dtype=np.float32)
        if not (isinstance(self.action_space, gym.spaces.Box) or isinstance(self.action_space, gym.spaces.Discrete)):
            raise ValueError("Unsupported action space")

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs.proprioception = np.concatenate((obs.proprioception, np.array([self.epi_time] * self.env.num_envs)))
        self.epi_time += 1.0 / self.time_limit
        return obs, rews, terminateds, truncateds, infos
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.epi_time = -0.5
        obs.proprioception = np.concatenate((obs.proprioception, np.array([self.epi_time])))
        return obs, info