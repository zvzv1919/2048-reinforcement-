import gym_2048
import gym
import numpy as np
import copy
import math

name = 'mcts model'

explore_width = 5
explore_depth = 15


class Model():
    def __init__(self):
        #TODO: make this LRU_cache
        self.mem = {}

    def predict(self, env):
        virtual_env = gym_2048.Game2048Env()
        greedy_action = 0
        greedy_score = float('-inf')
        for action in range(4):
            virtual_env.set_board(copy.deepcopy(env.get_board()))
            next_state, reward, done, info = virtual_env.step(action)
            score = evaluate_state(virtual_env)
            if score > greedy_score:
                greedy_score = score
                greedy_action = action
        # env.render()
        # print(greedy_action)
        return greedy_action

    def train(self, env):
        pass


def evaluate_state(env):
    # TODO: add heuristics
    virtual_env = gym_2048.Game2048Env()
    scores = []
    for i in range(explore_width):
        virtual_env.set_board(copy.deepcopy(env.get_board()))
        for j in range(explore_depth):
            # Randomly choose action in explore mode
            action = np.random.randint(0,4)
            next_state, _, done, info = virtual_env.step(action)
            if done:
                scores.append(virtual_env.steps)
                break
        scores.append(explore_depth * 0.5 + virtual_env.steps)
    return sum(scores) / len(scores) + max(scores)
#
# def mat_to_hash(matrix):
#     matrix = np.log(matrix)
#     hash = 0
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             hash += matrix[i][j]
#             hash *= 11
#             hash %= 10^9 + 7
#     return hash