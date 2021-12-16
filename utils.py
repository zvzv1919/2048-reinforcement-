import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def load_study(file_path):
    stat_dic = {}
    with open(file_path) as f:
        
        lines = f.readlines()
        for line in lines:
            if line.startswith( '|' ):
                temp = [x.strip() for x in line.split('|')]
                info_pair = [info for info in temp if info]
                k ,v  = info_pair[0], info_pair[1]
                try: 
                    v = float(v)
                except:
                    # percentage to float
                    v = float(v.strip('%'))/100
                if k not in stat_dic:  
                    stat_dic[k] = []
                stat_dic[k].append(v)
            else:
                pass
    return stat_dic

def stats(stat):
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.scatter(range(len(stat["episode score"])), stat["episode score"],marker = ",")
    ax1.set_title('Score vs Episode')
    ax2.scatter(range(len(stat["episode steps"])), stat["episode steps"],marker = ",")
    ax2.set_title('Number of steps played vs Episode')
    ax3.scatter(range(len(stat["highest"])), stat["highest"],marker = ",")
    ax3.set_title('Value of highest tile vs Episode')
    plt.show()