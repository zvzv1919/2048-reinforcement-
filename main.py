import gym_2048
import gym
import numpy as np
import randommodel

num_runs = 10
model = randommodel

if __name__ == '__main__':
  model_name = model.name
  model = model.Model()
  env = gym.make('2048-v0')
  model.train(env)
  # env.render()
  moves_arr = []

  for i in range(num_runs):
    env.reset()
    done = False
    moves = 0
    while not done:
      action = model.predict(env)
      next_state, reward, done, info = env.step(action)
      moves += 1

      # print('Next Action: "{}"\n\nReward: {}'.format(gym_2048.Base2048Env.ACTION_STRING[action], reward))
      # env.render()
    moves_arr.append(moves)

    print('\nTotal Moves: {}'.format(moves))
  print("Average Moves for {runs} runs with {model_name}: {moves}".format(model_name = model_name, runs = num_runs, moves = sum(moves_arr)/len(moves_arr)))