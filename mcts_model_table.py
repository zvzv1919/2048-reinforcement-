import gym_2048
import gym
import numpy as np
import copy
import math
import collections
from multiprocessing import Pool
import threading

name = 'mcts model table'

explore_width = 5
explore_depth = 15
learning_rate = 0.4
action_space = [0,1,2,3]

class Model():
    def __init__(self):
        #TODO: make this LRU_cache
        self.mem = collections.defaultdict(int)


    def predict(self, env):
        action_values = [0,0,0,0]
        lock = threading.Lock()
        thread_pool = []
        for action in action_space:
            thread_pool.append(threading.Thread(target = evaluate_action, args = (self.mem, env, action, action_values, lock)))
            thread_pool[-1].start()
        for t in thread_pool:
            t.join()
        # env.render()
        # print(action_values)
        return np.argmax(action_values)

    def train(self, env):
        pass

def evaluate_action(mem, env, action, action_values, lock):
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
        _, _, done, _ = virtual_env.step(action)
        while virtual_env.steps < explore_depth:
            if done:
                break
            state = mat_to_hash(virtual_env.get_board())

            # if legal move, append the new state to the state sequence
            if state != states[-1]:
                states.append(state)

            # Randomly choose successive action in explore mode
            explore_action = np.random.randint(0, 4)
            _, _, done, _ = virtual_env.step(explore_action)

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
    # print(action, scores)

    # lock.acquire()
    # print(sum(scores) / len(scores) + max(scores))
    # print("lock acquired, ", action_values)
    action_values[action] = sum(scores) / len(scores) + max(scores)
    # print("lock released, ", action_values)
    # lock.release()

def evaluate_state(env):
    # TODO: add heuristics
    virtual_env = gym_2048.Game2048Env()
    scores = []
    for i in range(explore_width):
        virtual_env.set_board(copy.deepcopy(env.get_board()))
        virtual_env.steps = 0

        while virtual_env.steps < explore_depth:
            # Randomly choose action in explore mode
            if done:
                # Only record number of legal steps
                scores.append(virtual_env.steps)
                break
            action = np.random.randint(0,4)
            next_state, _, done, info = virtual_env.step(action)

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
            hash %= 10**9 + 7
    return hash