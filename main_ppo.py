from gym_2048 import Game2048Env
import numpy as np
from ppo_agent import Agent
import logger
from utils import plot_learning_curve
from tqdm import tqdm

if __name__ == '__main__':
    # logger.configure(dir="./log/", format_strs="stdout,log")
    logger.configure(dir="./log/", format_strs="log")

    import datetime
    now = datetime.datetime.now()
    figure_name = f'game2048_{now.day}_{now.hour}_{now.minute}'
    figure_file = 'plots/' + figure_name

    chkpt_dit_name = f'tmp/ppo_17_20'
    N = 20
    env = Game2048Env()
    batch_size = 128
    n_epochs = 6
    alpha = 0.0003
    # alpha = 1e-4
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape[-1], chkpt_dir=chkpt_dit_name)
    agent.load_models()
    n_games = 300000
    log_iter = 10
    check_board = 20000
    

    best_score = env.reward_range[0]
    score_history = []
    step_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    Direction = {0:'up',1:'right',2:'down',3:'left'}

    for i in tqdm(range(n_games)):
        observation,reward, done, info = env.reset()
        num_steps = 0
        # score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            num_steps += 1
            n_steps += 1
            # score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            if (i % log_iter == 0) and done:
                logger.logkv('highest', info['highest'])
                logger.logkv('episode score',info['score'])
                logger.logkv('episode steps', info['steps'])
                logger.logkv('train progress',"{0:.0%}".format(i/n_games))
                logger.dumpkvs()
        step_history.append(num_steps)
        score_history.append(info['score'])
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()


        if (i % check_board == 0):
            print(f'Finished iteration {i}, the board end with:')
            print(observation)
            print(f'with last action {Direction[action]}')
            print(f'score: {score_history[-1]}')
            print('===================================')
        


    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    plot_learning_curve(x, step_history, figure_file + 'step')
    print(f'average steps played is {np.mean(step_history[-100:])}')
