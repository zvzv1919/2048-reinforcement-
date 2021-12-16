import gym_2048
import gym
import numpy as np
import copy
import math
import collections
from multiprocessing import Pool

name = 'mcts model table'

explore_width = 10
explore_depth = 30
learning_rate = 0.4
action_space = [0,1,2,3]

class Model():
    def __init__(self):
        #TODO: make this LRU_cache
        self.mem = collections.defaultdict(int)

    def predict(self, env):
        action_sequence = action_space
        with Pool(processes=len(action_sequence)) as pool:
            #TODO: Exploitation should consider the randomness, add exploitation depth
            action_values = pool.starmap(evaluate_action, [(self.mem, env, action) for action in action_sequence])
        return action_sequence[np.argmax(action_values)]

    def train(self, env):
        pass

def evaluate_action(mem, env, action):
    # print("jaja, {}".format(env.steps))
    virtual_env = gym_2048.Game2048Env()
    scores = []

    for i in range(explore_width):
        virtual_env.set_board(copy.deepcopy(env.get_board()))
        virtual_env.steps = 0
        states = [mat_to_hash(virtual_env.get_board())]
        done = False
        score = 0

        # The first action would be the one specified
        virtual_env.step(action)
        while virtual_env.steps < explore_depth:
            state = mat_to_hash(virtual_env.get_board())

            # if legal move, append the new state to the state sequence
            if state != states[-1]:
                states.append(state)

            # Randomly choose successive action in explore mode
            action = np.random.randint(0, 4)
            _, _, done, _ = virtual_env.step(action)
            if done:
                break

        score = mem[mat_to_hash(virtual_env.get_board())]
        if not done and score == 0:
            # initialize current state score to non-zero value
            mem[mat_to_hash(virtual_env.get_board())]=explore_depth*0.3
            score = explore_depth*0.3

        # Back propagates reward to all states visited in this trial
        for state in states[::-1]:
            mem[state] += learning_rate * (score - mem[state])
            score += 1

        # Trial over, use the score of first state in this trial as the trial score
        scores.append(mem[states[0]])

    return sum(scores) / len(scores) + max(scores)

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

def mat_to_hash(matrix):
    hash = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                hash += 0
            else:
                hash += int(math.log2(matrix[i][j])) + 1
            hash *= 13
            hash %= 10^9 + 7
    return hash