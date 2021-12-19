import gym_2048
import gym
import numpy as np
import copy
import math
import collections
from multiprocessing import Pool
import threading
import json

name = 'mcts model table'

explore_width = 20
explore_depth = 35
training_episode = 500
learning_rate = 0.2
action_space = [0,1,2,3]

class Model():
    def __init__(self):
        #TODO: make this LRU_cache
        self.mem = collections.defaultdict(int)
        self.N = collections.defaultdict(int)

    def predict(self, env):
        action_values = [0,0,0,0]
        thread_pool = []
        for action in action_space:
            thread_pool.append(threading.Thread(target = evaluate_action, args = (self.mem, self.N, env, action, action_values)))
            thread_pool[-1].start()
        for t in thread_pool:
            t.join()
        # env.render()
        # print(action_values)

        return np.argmax(action_values)

    def train(self, env, load = None, save = None):
        if load is not None:
            self.load_model(load)

        moves_arr = []
        score_arr = []
        for i in range(training_episode):
            env.reset()
            done = False
            while not done:
                action = self.predict(env)
                next_state, reward, done, info = env.step(action)

                # print('Next Action: "{}"\n\nReward: {}'.format(gym_2048.Base2048Env.ACTION_STRING[action], reward))
                # env.render()
            moves_arr.append(env.steps)
            score_arr.append(env.score)
            # env.render()
            print("training progress: {i}/{training_episode}".format(i=i+1, training_episode=training_episode))

        if save is not None:
            f = open("training_log:{}".format(save), "w")
            json.dump({"moves_arr":moves_arr, "score_arr":score_arr}, f)
            f.close()
            self.model_save(save)
        pass

    def model_save(self, file_name):
        f = open("{}.json".format(file_name), 'w')
        json.dump(self.mem, f)
        f.close()

    def load_model(self, file_name):
        f = open("{}.json".format(file_name), 'r')
        self.mem = collections.defaultdict(int, json.load(f))
        f.close()

def evaluate_action(mem, N, env, action, action_values):
    # print("jaja, {}".format(env.steps))
    virtual_env = gym_2048.Game2048Env()
    scores = []

    for i in range(explore_width):
        virtual_env.set_board(copy.deepcopy(env.get_board()))
        virtual_env.steps = 0
        states = [mat_to_hash(virtual_env.get_board())]
        done = False
        score = 0
        illegal_actions = set() #Used to remember illegal actions for the current exploration step

        # The first action would be the one specified
        next_state, reward, done, _ = virtual_env.step(action)

        # If action is illegal, set value to 0
        if reward == -1:
            action_values[action] = 0
            return

        while virtual_env.steps <= explore_depth and not done:

            state = mat_to_hash(next_state)
            # if legal move, append the new state to the state sequence
            if state != states[-1]:
                states.append(state)

            # Randomly choose successive action in explore mode
            explore_action = np.random.randint(0, 4)
            # If the previous move is illegal, try to choose between legal moves
            if reward == virtual_env.illegal_move_reward:
                while explore_action in illegal_actions:
                    explore_action = np.random.randint(0, 4)

            next_state, reward, done, _ = virtual_env.step(explore_action)
            if reward == env.illegal_move_reward:
                illegal_actions.add(explore_action)
            else:
                illegal_actions = set()

        score = mem[mat_to_hash(next_state)]
        # if not done and score == 0:
        #     # initialize current state score to non-zero value
        #     mem[mat_to_hash(next_state)] = 3
        #     score = 3

        # Back propagates reward to all states visited in this trial
        for state in states[::-1]:
            score += 1
            mem[state] = (mem[state] * N[state] + score) / (N[state] + 1)
            N[state] += 1
            # if mem[state] == 0:
            #     mem[state] = score
            # else:
            #     mem[state] = mem[state] + learning_rate * (score - mem[state])

        # Trial over, use the score of first state in this trial as the trial score
        if len(states) == 1:
            scores.append(1)
        else:
            scores.append(1 + mem[states[1]])
    # print(action, scores)

    action_values[action] = sum(scores) / len(scores)


# def evaluate_state(env):
#     # TODO: add heuristics
#     virtual_env = gym_2048.Game2048Env()
#     scores = []
#     for i in range(explore_width):
#         virtual_env.set_board(copy.deepcopy(env.get_board()))
#         virtual_env.steps = 0
#
#         while virtual_env.steps < explore_depth:
#             # Randomly choose action in explore mode
#             if done:
#                 # Only record number of legal steps
#                 scores.append(virtual_env.steps)
#                 break
#             action = np.random.randint(0,4)
#             next_state, _, done, info = virtual_env.step(action)
#
#         # Add additional reward if the trail does not meet end state
#         if len(scores) <= i:
#             scores.append(explore_depth * 0.5 + virtual_env.steps)
#     return sum(scores) / len(scores)

def mat_to_hash(matrix):
    hash = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            hash *= 10
            if matrix[i][j] == 0:
                hash += 0
            else:
                hash += int(math.log2(matrix[i][j])) % 10
    return hash