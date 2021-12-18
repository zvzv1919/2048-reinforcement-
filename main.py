from gym_2048 import Game2048Env
import gym
import numpy as np
import randommodel
import greedy_model
import mcts_model_vanilla
import mcts_model_table
import dqn_model
import matplotlib.pyplot as plt




num_runs = 50
model = dqn_model

def plot_perf(data):
    fig, ax = plt.subplots()

    x = np.linspace(0,num_runs, num_runs)
    ax.plot(x, data, linewidth=2.0)

    ax.set_xlabel('run')
    ax.set_ylabel('moves')
    plt.show()

if __name__ == '__main__':
  model_name = model.name
  env = Game2048Env()
  model = model.Model(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[env.observation_space.shape[-1]], lr=0.001)
  
  # model.train(env, load="model 3M", save="model 5M")

  # env.render()
  moves_arr = []
  score_arr = []

  for i in range(num_runs):
    observation,_,_,_  = env.reset()
    observation = observation.reshape(-1)
    done = False
    while not done:
      action = model.predict(observation)
      observation_, reward, done, info = env.step(action)
      observation_ = observation_.reshape(-1)
      model.store_transition(observation, action, reward,
                                    observation_, done)
      model.learn()
      observation = observation_      
      # print('Next Action: "{}"\n\nReward: {}'.format(gym_2048.Base2048Env.ACTION_STRING[action], reward))
      # env.render()
    moves_arr.append(env.steps)
    score_arr.append(env.score)
    env.render()

    print('\nTotal Moves: {}'.format(moves_arr[-1]))
    print(np.max(observation_))
    # print("mem len:", len(model.mem))
  print("Average Moves for {runs} runs with {model_name}: {moves}".format(model_name = model_name, runs = num_runs, moves = sum(moves_arr)/len(moves_arr)))
  print("Average Score for {runs} runs with {model_name}: {moves}".format(model_name = model_name, runs = num_runs, moves = sum(score_arr)/len(score_arr)))
  plot_perf(moves_arr)




