import matplotlib.pyplot as plt
import numpy as np
import math
import os


def average_over_several_runs(folder):
    data_all = []
    min_length = np.inf
    runs = os.listdir(folder)
    for i in range(len(runs)):
        data = np.loadtxt(folder+'/'+runs[i]+'/log.csv', delimiter=',', dtype=str)
        evaluation_freq = float(data[3, 1])-float(data[1, 1])
        episode_reward = (data[:, 4][data[:, 4] != 'rreturn_mean']).astype(float)
        data_all.append(episode_reward)
        if episode_reward.shape[0] < min_length:
            min_length = episode_reward.shape[0]
    average = np.zeros([len(runs), min_length])
    for i in range(len(runs)):
        average[i, :] = data_all[i][:min_length]
    mean = np.mean(average, axis=0)
    std = np.std(average, axis=0)

    return mean, std, evaluation_freq/1000


def plot_several_folders(prefix, folders, period=0, label_list=[], plot_or_save='save', title=""):
    plt.rcParams["figure.figsize"] = (6, 5)
    # plt.rcParams["figure.figsize"] = (15, 12)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        folder_name = 'saved_exps/'+prefix+folders[i]
        num_runs = len(os.listdir(folder_name))
        mean_all, std_all, eval_freq = average_over_several_runs(folder_name)

        # plot variance
        axs.fill_between(eval_freq * np.arange(len(mean_all)),
                mean_all - std_all/math.sqrt(num_runs),
                mean_all + std_all/math.sqrt(num_runs), alpha=0.4)
        if len(label_list) == len(folders):
            # specify label
            axs.plot(eval_freq * np.arange(len(mean_all)), mean_all, label=label_list[i])
        else:
            axs.plot(eval_freq * np.arange(len(mean_all)), mean_all, label=folders[i])

        axs.set_xlabel('frames(x1000)')
        axs.set_ylabel('episode reward')
        axs.legend(fontsize=10)
        # axs_plot.set_title(eval_env_type[j])
        axs.set_title(title)
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_figs/'+title)

# 3.21
prefix = 'doorkey-5x5-v0/'
folders_1 = ['ppo']
plot_several_folders(prefix, folders_1, title='doorkey-5x5')