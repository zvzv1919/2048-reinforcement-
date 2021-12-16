import gym_2048
import gym
import numpy as np
import copy
import math
import collections
from multiprocessing import Pool

name = 'mcts model vanilla'

explore_width = 10
explore_depth = 30
action_space = [0,1,2,3]

class Model():
    def __init__(self):
        #TODO: make this LRU_cache
        self.mem = collections.defaultdict(int)

    def predict(self, env):
        action_sequence = action_space
        with Pool(processes=len(action_sequence)) as pool:
            #TODO: Exploitation should consider the randomness, add exploitation depth
            action_values = pool.starmap(evaluate_action, [(env, action) for action in action_sequence])
        return action_sequence[np.argmax(action_values)]

    def train(self, env):
        pass

def evaluate_action(env, action):
    # print("jaja, {}".format(env.steps))
    virtual_env = gym_2048.Game2048Env()
    virtual_env.set_board(copy.deepcopy(env.get_board()))
    virtual_env.step(action)
    return evaluate_state(virtual_env)

def evaluate_state(env):
    # TODO: add heuristics
    virtual_env = gym_2048.Game2048Env()
    scores = []
    for i in range(explore_width):
        virtual_env.set_board(copy.deepcopy(env.get_board()))
        virtual_env.steps = 0

        while virtual_env.steps < explore_depth:
            # Randomly choose action in explore mode
            action = np.random.randint(0,4)
            next_state, _, done, info = virtual_env.step(action)
            if done:
                # Only record number of legal steps
                scores.append(virtual_env.steps)
                break
        # Add additional reward if the trail does not meet end state
        if len(scores) <= i:
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