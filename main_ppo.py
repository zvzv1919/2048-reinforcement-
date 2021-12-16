from gym_2048 import Game2048Env
import numpy as np
from ppo_agent import Agent
import logger
from utils import plot_learning_curve

if __name__ == '__main__':
    # logger.configure(dir="./log/", format_strs="stdout,log")
    logger.configure(dir="./log/", format_strs="log")

    env = Game2048Env()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape[-1])
    n_games = 20000
    log_iter = 10
    check_board = 30
    import datetime
    now = datetime.datetime.now()
    figure_name = f'game2048_{now.day}_{now.hour}_{now.minute}'
    figure_file = 'plots/' + figure_name

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    Direction = {0:'up',1:'right',2:'down',3:'left'}

    for i in range(n_games):
        observation,reward, done, info = env.reset()
        # score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            # score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            if (i % log_iter == 0) and done:
                logger.logkv('illegal move', info['num_illegal'])
                logger.logkv('highest', info['highest'])
                logger.logkv('episode score',info['score'])
                logger.logkv('episode steps', info['steps'])
                logger.logkv('train progress',"{0:.0%}".format(i/n_games))
                logger.dumpkvs()
        score_history.append(info['score'])
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()


        if (i % check_board == 0):
            print(f'Finished iteration {i}, the board end with:')
            print(observation)
            print(f'with last action {Direction[action]}')
            print('===================================')
        


    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

