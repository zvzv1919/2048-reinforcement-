import gym_2048
import gym
import numpy as np
import copy
import math

name = 'greedy model'

class Model():

    def predict(self, env):
        virtual_env = gym_2048.Game2048Env()
        greedy_action = 0
        greedy_reward = float('-inf')
        for action in range(4):
            virtual_env.set_board(copy.deepcopy(env.get_board()))
            next_state, reward, done, info = virtual_env.step(action)

            if reward > greedy_reward:
                greedy_reward = reward
                greedy_action = action
        # env.render()
        # print(greedy_action)
        return greedy_action

    def train(self, env):
        pass

