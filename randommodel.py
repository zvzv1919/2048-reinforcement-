import gym_2048
import gym
import numpy as np

name = 'random model'
class Model():
    def predict(self, env):
        return env.np_random.choice(range(4), 1).item()
    def train(self, env):
        pass