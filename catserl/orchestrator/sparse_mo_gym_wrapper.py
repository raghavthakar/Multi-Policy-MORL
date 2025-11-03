import gymnasium as gym
import numpy as np

class SparseMultiObjectiveRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reward_dim = None
        self.cumulative_reward = None
        self.steps_since_last_reset = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_dim = None
        self.cumulative_reward = None
        self.steps_since_last_reset = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.reward_dim is None:
            self.reward_dim = len(reward)
            self.cumulative_reward = np.zeros(self.reward_dim, dtype=np.float32)

        self.cumulative_reward += reward
        self.steps_since_last_reset += 1
        done = terminated or truncated

        if done:
            averaged_reward = self.cumulative_reward / max(1, self.steps_since_last_reset)
            sparse_reward = averaged_reward
            self.cumulative_reward[:] = 0.0
            self.steps_since_last_reset = 0
        else:
            sparse_reward = np.zeros(self.reward_dim, dtype=np.float32)

        return obs, sparse_reward, terminated, truncated, info
