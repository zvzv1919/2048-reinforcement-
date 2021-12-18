from gym_2048 import Game2048Env
import gym
import numpy as np
import randommodel
import greedy_model
import mcts_model_vanilla
import mcts_model_table
import matplotlib.pyplot as plt




num_runs = 50
model = mcts_model_table

def plot_perf(data):
    fig, ax = plt.subplots()

    x = np.linspace(0,num_runs, num_runs)
    ax.plot(x, data, linewidth=2.0)

    ax.set_xlabel('run')
    ax.set_ylabel('moves')
    plt.show()

if __name__ == '__main__':
  model_name = model.name
  model = model.Model()
  env = Game2048Env()
  model.train(env, load="model 7M", save="model 9M")

  # env.render()
  moves_arr = []
  score_arr = []

  for i in range(num_runs):
    env.reset()
    done = False
    while not done:
      action = model.predict(env)
      next_state, reward, done, info = env.step(action)

      # print('Next Action: "{}"\n\nReward: {}'.format(gym_2048.Base2048Env.ACTION_STRING[action], reward))
      # env.render()
    moves_arr.append(env.steps)
    score_arr.append(env.score)
    env.render()

    print('\nTotal Moves: {}'.format(moves_arr[-1]))
    print(np.max(next_state))
    print("mem len:", len(model.mem))
  print("Average Moves for {runs} runs with {model_name}: {moves}".format(model_name = model_name, runs = num_runs, moves = sum(moves_arr)/len(moves_arr)))
  print("Average Score for {runs} runs with {model_name}: {moves}".format(model_name = model_name, runs = num_runs, moves = sum(score_arr)/len(score_arr)))
  plot_perf(moves_arr)



