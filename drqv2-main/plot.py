import matplotlib.pyplot as plt
import numpy as np
import math
import os


def average_over_several_runs(folder, smooth):
    data_all = []
    min_length = np.inf
    runs = os.listdir(folder)
    for i in range(len(runs)):
        data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
        evaluation_freq = data[2, -3]-data[1, -3]
        if smooth:
            data_all.append(data[:, 3])
        else:
            data_all.append(data[:, 2])
        if data.shape[0] < min_length:
            min_length = data.shape[0]
    average = np.zeros([len(runs), min_length])
    for i in range(len(runs)):
        average[i, :] = data_all[i][:min_length]
    mean = np.mean(average, axis=0)
    std = np.std(average, axis=0)

    return mean, std, evaluation_freq/1000


def plot_several_folders(prefix, folders, period=0, label_list=[], plot_or_save='save', title="", smooth=False):
    plt.rcParams["figure.figsize"] = (6, 5)
    # plt.rcParams["figure.figsize"] = (15, 12)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        if isinstance(folders[i], list):
            # action repeat 2
            mean_1, std_1, eval_freq_1 = average_over_several_runs('saved_exps/' + prefix + folders[i][0])
            iters_1 = eval_freq_1 * np.arange(len(mean_1))
            # action repeat 1
            mean_2, std_2, eval_freq_2 = average_over_several_runs('saved_exps/' + prefix + folders[i][1])
            iters_2 = eval_freq_2 * np.arange(len(mean_2))

            iters_all = np.concatenate([iters_1[0:period[i]], iters_2+iters_1[period[i]]])
            mean_all = np.concatenate([mean_1[0:period[i]], mean_2])
            std_all = np.concatenate([std_1[0:period[i]], std_2])

            # plot variance
            axs.fill_between(iters_all,
                    mean_all - std_all/math.sqrt(num_runs),
                    mean_all + std_all/math.sqrt(num_runs), alpha=0.4)
            if len(label_list) == len(folders):
                # specify label
                axs.plot(iters_all, mean_all, label=label_list[i])
            else:
                axs.plot(iters_all, mean_all, label=folders[i][-1])
        else:
            folder_name = 'saved_exps/'+prefix+folders[i]
            num_runs = len(os.listdir(folder_name))
            mean_all, std_all, eval_freq = average_over_several_runs(folder_name, smooth)

            if prefix not in ['humanoid_stand/', 'humanoid_walk/', 'humanoid_run/']:
                if len(mean_all) > 100:
                    mean_all = mean_all[:100]
                    std_all = std_all[:100]

            # plot variance
            axs.fill_between(eval_freq * range(len(mean_all)),
                    mean_all - std_all/math.sqrt(num_runs),
                    mean_all + std_all/math.sqrt(num_runs), alpha=0.4)
            if len(label_list) == len(folders):
                # specify label
                axs.plot(eval_freq * range(len(mean_all)), mean_all, label=label_list[i])
            else:
                axs.plot(eval_freq * range(len(mean_all)), mean_all, label=folders[i])

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
prefix = 'acrobot_swingup/'
folders_1 = ['drqv2_epsilon_greedy', 'drqv2_epsilon_greedy_simhash_repeat', 'drqv2_epsilon_zeta',
             'drqv2_action_repeat_1_epsilon_greedy_simhash_repeat']
plot_several_folders(prefix, folders_1, title='acrobot_swingup_epsilon')

prefix = 'reacher_hard/'
folders_1 = ['drqv2_epsilon_greedy', 'drqv2_epsilon_greedy_simhash_repeat', 'drqv2_epsilon_zeta',
             'drqv2_action_repeat_1_epsilon_greedy_simhash_repeat']
plot_several_folders(prefix, folders_1, title='reacher_hard_epsilon')

prefix = 'quadruped_run/'
folders_1 = ['drqv2_epsilon_greedy', 'drqv2_epsilon_greedy_simhash_repeat', 'drqv2_epsilon_zeta',
             'drqv2_action_repeat_1_epsilon_greedy_simhash_repeat']
plot_several_folders(prefix, folders_1, title='quadruped_run_epsilon')

prefix = 'acrobot_swingup/'
folders_1 = ['drqv2_epsilon_greedy', 'drqv2_epsilon_greedy_simhash_repeat', 'drqv2_epsilon_zeta',
             'drqv2_action_repeat_1_epsilon_greedy_simhash_repeat']
plot_several_folders(prefix, folders_1, title='acrobot_swingup_epsilon_smoothness', smooth=True)

prefix = 'reacher_hard/'
folders_1 = ['drqv2_epsilon_greedy', 'drqv2_epsilon_greedy_simhash_repeat', 'drqv2_epsilon_zeta',
             'drqv2_action_repeat_1_epsilon_greedy_simhash_repeat']
plot_several_folders(prefix, folders_1, title='reacher_hard_epsilon_smoothness', smooth=True)

prefix = 'quadruped_run/'
folders_1 = ['drqv2_epsilon_greedy', 'drqv2_epsilon_greedy_simhash_repeat', 'drqv2_epsilon_zeta',
             'drqv2_action_repeat_1_epsilon_greedy_simhash_repeat']
plot_several_folders(prefix, folders_1, title='quadruped_run_epsilon_smoothness', smooth=True)


prefix = 'reacher_hard/'
folders_1 = ['drqv2_noisy_net_critic', 'drqv2_noisy_net_actor', 'drqv2_noisy_net_actor_critic']
plot_several_folders(prefix, folders_1, title='reacher_hard_noisy_net')



# 3.7
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_repeat_2_simhash_count', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, title='quadruped_run_sim_count')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_repeat_2_simhash_count', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup_sim_count')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_repeat_2_simhash_count', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, title='reacher_hard_sim_count')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2_sac', 'drqv2_sac_simhash_count', 'drqv2_sac_repeat_1_simhash_count']
# plot_several_folders(prefix, folders_1, title='quadruped_run_sac_sim_count')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2_sac', 'drqv2_sac_simhash_count', 'drqv2_sac_repeat_1_simhash_count']
# plot_several_folders(prefix, folders_1, title='acroboot_swingup_sac_sim_count')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2_sac', 'drqv2_sac_simhash_count', 'drqv2_sac_repeat_1_simhash_count']
# plot_several_folders(prefix, folders_1, title='reacher_hard_sac_sim_count')

# 2.29
# period = [0, 0, 50, 50, 50, 50]
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4', 'drqv2_state_action_batch_unvisit']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_more_count')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4', 'drqv2_state_action_batch_unvisit']
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_more_count')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4', 'drqv2_state_action_batch_unvisit']
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_more_count')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_rnd_initial_10_loss_norm_obs', 'drqv2_rnd_initial_20_loss_norm_obs',
#              'drqv2_rnd_initial_20_loss_norm_obs_lr01', 'drqv2_rnd_initial_20_loss_norm_obs_lr001',
#              'drqv2_rnd_initial_20_loss_norm_obs_lr01_add_two_layer_predictor']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_rnd')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_rnd_initial_20_loss_norm_obs_lr01']
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_rnd')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_rnd_initial_20_loss_norm_obs_lr01']
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_rnd')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_ensemble_5_c1', 'drqv2_ensemble_5_c5', 'drqv2_ensemble_5_c25']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_uncertainty')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_uniform_01_sample_10_c01', 'drqv2_uniform_02_sample_10_c01']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_sample_uncertainty')


# 2.23
# period = [0, 0, 50, 50, 50, 50]
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_223pre')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_223pre')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_223pre')
#
# prefix = 'cheetah_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='cheetah_run_223pre')
#
# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_223pre')
#
# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_223pre')
#
# prefix = 'finger_spin/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='finger_spin_223pre')
#
# prefix = 'finger_turn_hard/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='finger_turn_hard_223pre')
#
# prefix = 'humanoid_stand/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='humanoid_stand_223pre')
#
# prefix = 'humanoid_walk/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='humanoid_walk_223pre')
#
# prefix = 'humanoid_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='humanoid_run_223pre')

# 2.22
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4',
#              'drqv2_repeat_2_to_1_500k_nstep6_upevery4',
#              'drqv2_unvisit_repeat_nstep6_upevery4', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_without_dyn')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4',
#              'drqv2_repeat_2_to_1_500k_nstep6_upevery4',
#              'drqv2_unvisit_repeat_nstep6_upevery4', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_without_dyn')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4',
#              'drqv2_repeat_2_to_1_500k_nstep6_upevery4',
#              'drqv2_unvisit_repeat_nstep6_upevery4', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_without_dyn')
#
# prefix = 'cheetah_run/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4','drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='cheetah_run_without_dyn')
#
# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4','drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_without_dyn')
#
# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4','drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_without_dyn')
#
# prefix = 'finger_spin/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4','drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='finger_spin_without_dyn')
#
# prefix = 'finger_turn_hard/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4','drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='finger_turn_hard_without_dyn')
#
# prefix = 'humanoid_stand/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='humanoid_stand_without_dyn')
#
# prefix = 'humanoid_walk/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='humanoid_walk_without_dyn')
#
# prefix = 'humanoid_run/'
# folders_1 = ['drqv2', 'drqv2_repeat_1_nstep6_upevery4', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='humanoid_run_without_dyn')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_3_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_4_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_time_ssl')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_3_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_4_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_time_ssl')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_3_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_timessl_4_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_time_ssl')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_proj_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_mlp_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_mlp_encoder_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_c2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_trainable_hash')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_proj_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_mlp_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_mlp_encoder_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_c2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_trainable_hash')
#
# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_proj_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_mlp_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_trainable_mlp_encoder_batch_unvisit_repeat_nstep6_upevery4',
#              'drqv2_c2_batch_unvisit_repeat_nstep6_upevery4']
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_trainable_hash')


# 2.8
# prefix = 'quadruped_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6_1e6',
#              'repeat_2_to_1_200k_half_obs', 'repeat_2_to_1_200k_initial_random']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_half_obs')

# prefix = 'acrobot_swingup/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_unvisit_long']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_unvisit')

# prefix = 'quadruped_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_unvisit_long']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_unvisit')

# # 2.1
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_nstep6')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_nstep6')
#
# prefix = 'cheetah_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='cheetah_run_nstep6')
#
# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_nstep6')
#
# prefix = 'hopper_hop/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_nstep6')
#
# prefix = 'quadruped_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6_1e6']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_nstep6')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6',
#              'repeat_geometric_05_decaying_200k', 'repeat_zeta_2', 'repeat_uncertain_last',
#              'repeat_uncertain_last_ensemble5', 'repeat_unvisit']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_random_repeat')
#
# prefix = 'quadruped_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k_nstep6_6_1e6',
#              'repeat_geometric_05_decaying_200k', 'repeat_zeta_2', 'repeat_uncertain_last',
#              'repeat_uncertain_last_ensemble5', 'repeat_unvisit']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_random_repeat')

# 1.25
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k',
#              'repeat_2_to_1_200k_reset',
#              'repeat_2_to_1_200k_ensemble_5_reset_up4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_action_repeat_1')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k', 'repeat_2_to_1_400k',
#              'repeat_2_to_1_200k_ensemble_5_reset_up4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_action_repeat_1')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k',
#              'repeat_2_to_1_200k_nstep2', 'repeat_4_to_2_200k_nstep4_2']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_action_repeat_4_2')
#
# prefix = 'cheetah_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k',
#              'repeat_2_to_1_200k_reset',
#              'repeat_2_to_1_200k_ensemble_5_reset_up4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='cheetah_run_action_repeat_1')
#
# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k', 'repeat_2_to_1_200k_reset',
#              'repeat_2_to_1_200k_ensemble_5_reset_up4']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_action_repeat_1')
#
# prefix = 'hopper_hop/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k', 'repeat_2_to_1_200k_reset']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_action_repeat_1')
#
# prefix = 'quadruped_run/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_1', 'repeat_2_to_1_200k']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_action_repeat_1')

# 1.18
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_nstep1'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_nstep1_half_old'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_nstep1_decaying_old']
#              ]
# period = [0, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_nstep_1')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2', 'action_repeat_2_update_4', 'action_repeat_2_ensemble_5_update_10']
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_update_more')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_nstep1_decaying_old'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_nstep1_decaying_old_update_10'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_calQL'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_calQL_200k_buffer']]
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_buffer')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2', 'action_repeat_2_update_4', 'action_repeat_2_ensemble_5_update_10']
# period = [0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_update_more')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['action_repeat_2_ensemble_5_update_10',
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_5e4_buffer_nstep1_decaying_old_q5up10'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_5e4_buffer_nstep1_decaying_old_q5up10'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_2500_bc'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_bc'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_normal'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_cql'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_calQL']
#              ]
# period = [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_update_more_transfer')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['action_repeat_2_ensemble_5_update_10',
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_calQL'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_25000_calQL'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_calQL_up4'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_calQL_noise'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_pretrain_5000_calQL_noise'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_calQL_no_decay'],
#              ['action_repeat_2_ensemble_5_update_10', 'q5up10_load_pretrain_5000_calQL_half_up4']
#              ]
# period = [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_calql')
#
#
# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2',
#              'baseline_action_repeat_2_ensemble_5_update_4']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_update_more')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['pretrain_test_load', 'pretrain_test_no_load']
# period = [0, 0, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_pretrain_test')


# 1.11
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_frames_pretrain_2500'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_frames_pretrain_2500_extend'],
#              ]
# period = [0, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_pretrain')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_update_4'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_update_8'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_1e5_buffer'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_1e5_buffer_update_4'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_more_explore']
#              ]
# period = [0, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_buffer')
#
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_3Q'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_5Q'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_5e4_buffer_5Q_up4']
#              ]
# period = [0, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_buffer_ensemble')


# 1.4
# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'baseline', 'dyn_model_3', 'dyn_model_3_only_linear', 'dyn_model_3_linear_nonlinear']
# period = [0, 0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_dyn_model')

# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'baseline', 'baseline_action_repeat_2_update_4']
# period = [0, 0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_update_more')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_time_scale_4',
#              'baseline_action_repeat_2_time_ssl_K_4_critic',
#              'baseline_action_repeat_4',
#              'baseline_action_repeat_4_time_ssl_K_4_actor',
#              'baseline_action_repeat_4_time_ssl_K_4_critic',
#              'baseline_action_repeat_4_time_ssl_K_3_critic']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup_scale')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2', 'baseline_action_repeat_2_time_ssl_K_4_critic',
# ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_new_policy'],
#              ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_time_ssl_K_4_critic'],
#              ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_update_more'],
#              ['baseline_action_repeat_2_time_ssl_K_4_critic',
#               'baseline_action_repeat_1_load_2_encoder_only_old_policy_always_update_more'],
#              ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_old_policy_always_update_more'],
#              ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_new_policy_always_update_more']
#              ]
# period = [0, 0, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_scale_load_1_2')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old'],
#              ['baseline_action_repeat_2', 'action_repeat_1_decaying_old'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old_099995'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old_099999'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old_obs']
#              ]
# period = [0, 50, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_1_load_2')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old_longer']
#              ]
# period = [0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_1_load_2_longer')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old'],
#              ['baseline_action_repeat_2', 'action_repeat_1_decaying_old'],
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old_update_4'],
#              ['baseline_action_repeat_2', 'action_repeat_1_decaying_old_update_4'],
#              ['baseline_action_repeat_2', 'action_repeat_1_decaying_old_update_8'],
#              ['baseline_action_repeat_2', 'action_repeat_1_decaying_old_update_16']
#              ]
# period = [0, 50, 50, 50, 50, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_1_load_2_update_more')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_4',
#              ['baseline_action_repeat_4', 'baseline_action_repeat_2_load_4'],
#              ['baseline_action_repeat_4', 'baseline_action_repeat_2_load_4_update_more']]
# period = [0, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_scale_load_2_4')
#
# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2', 'action_repeat_2_add_last_action',
#              'action_repeat_2_frame_stack_2_add_last_action',
#              'action_repeat_2_frame_stack_2_add_last_action_process_each_frame']
# period = [0, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_add_last_action')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['baseline_action_repeat_2', 'action_repeat_2_add_last_action',
#              'action_repeat_2_frame_stack_2_add_last_action',
#              'action_repeat_2_frame_stack_2_add_last_action_process_each_frame']
# period = [0, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_add_last_action')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old']]
# period = [0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_load')
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'action_repeat_1_load_2_decaying_old_longer']]
# period = [0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_load_longer')
#
# prefix = 'cheetah_run/'
# folders_1 = ['drqv2', 'baseline',
#              # ['baseline', 'baseline_action_repeat_1_load_2'],
#              ['baseline', 'action_repeat_1_load_2_decaying_old']]
# period = [0, 0, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='cheetah_run_load')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'baseline',
#              # ['baseline', 'baseline_action_repeat_1_load_2'],
#              ['baseline', 'action_repeat_1_load_2_decaying_old']]
# period = [0, 0, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_load')


# prefix = 'reach_duplo/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'baseline_action_repeat_1_load_repeat_2'],
#              ['baseline_action_repeat_2', 'baseline_action_repeat_1_load_repeat_2_step_125000']]
# period = [0, 50, 25]
# plot_several_folders(prefix, folders_1, period=period, title='reach_duplo_same_frames')

# prefix = 'cheetah_run/'
# folders_1 = ['drqv2', 'baseline', 'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='cheetah_run_dyn_prior')

# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'baseline', 'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_dyn_prior')

# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'baseline', 'baseline_action_repeat_2_time_ssl_K_4_critic',
#              ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_time_ssl_K_4_critic']]
# period = [0, 0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='hopper_hop_scale_load')


# prefix = 'fish_swim/'
# folders_1 = ['baseline', 'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='fish_swim_dyn_prior')
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'baseline', 'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='quadruped_run_dyn_prior')

# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2',
#              'baseline_action_repeat_2_two_steps_invariance']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_two_steps_invariance')

# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl',
#              'baseline_action_repeat_2_time_ssl_perturb_3', 'baseline_action_repeat_2_time_ssl_perturb_4',
#              'baseline_action_repeat_2_time_ssl_perturb_3_trainable_temp']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_ssl')

# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_backward_1',
#              'baseline_action_repeat_2_time_ssl_backward_1_trainable_temp',
#              'baseline_action_repeat_2_time_ssl_backward_2',
#              'baseline_action_repeat_2_time_ssl_backward_3',
#              'baseline_action_repeat_2_time_ssl_backward_1_random_last_two_frames']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_backward')


# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'baseline_action_repeat_1_load_repeat_2'],
#              'baseline_action_repeat_2_time_ssl_scale_4',
#              ['baseline_action_repeat_2_time_ssl_scale_4', 'baseline_action_repeat_1_load_repeat_2_time_ssl_time_scale_4']
#              ]
# period = [0, 0, 50, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_same_frames')

# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_K_4_critic',
#              ['baseline_action_repeat_2_time_ssl_K_4_critic', 'baseline_action_repeat_1_load_2_time_ssl_K_4_critic']]
# period = [0, 0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_scale_load')

# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_dyn_prior_5_previous_obs',
#              'baseline_action_repeat_2_dyn_prior_9_previous_obs',
#              'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss',
#              'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss_15_states',
#              'baseline_action_repeat_2_dyn_prior_5_time_ssl_4']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_dyn_prior')

# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_scale_3',
#              'baseline_action_repeat_2_time_ssl_scale_4',
#              'baseline_action_repeat_2_time_ssl_scale_5',
#              'baseline_action_repeat_2_time_ssl_scale_4_rank_k',
#              'baseline_action_repeat_2_time_ssl_K_4_critic']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_scale')

# prefix = 'walker_run/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'inv_dyn_model',
#              'inv_dyn_model_reflected_obs_loss',
#              'inv_dyn_model_reflected_obs_loss_time_ssl_time_scale_4',
#              'inv_dyn_model_reverse_loss_wo_reflect',
#              'inv_dyn_model_true_reflected_loss']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_inv_dyn')


# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              'baseline_action_repeat_2_two_steps_invariance',
#              'baseline_action_repeat_2_two_steps_invariance_weight_1']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_two_steps_invariance')

# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_time_scale_4']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_scale')

# prefix = 'reacher_hard/'
# folders_1 = ['baseline_action_repeat_2', 'dyn_model_3', 'dyn_model_3_only_linear']
# period = [0, 0]
# plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_dyn_model')

# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'inv_dyn_model_reverse_loss_wo_reflect']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup_inv_dyn')


# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_time_scale_4',
#              ['baseline_action_repeat_2_time_ssl_time_scale_4', 'baseline_action_repeat_1_load_2_time_ssl_time_scale_4'],
#              ['baseline_action_repeat_2_time_ssl_time_scale_4', 'baseline_action_repeat_1_load_2_step_375000_time_ssl_time_scale_4'],
#             ['baseline_action_repeat_2_time_ssl_time_scale_4', 'baseline_action_repeat_1_load_2_add_bc_loss_repeat'],
#             ['baseline_action_repeat_2_time_ssl_time_scale_4', 'baseline_action_repeat_1_load_2_add_bc_loss_repeat_weight_1'],
#              ['baseline_action_repeat_2_time_ssl_time_scale_4', 'baseline_action_repeat_1_load_2_choose_action_bc']
#              ]
# period = [0, 0, 0, 50, 75, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_same_frames')

# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_4',
#              'baseline_action_repeat_4_time_ssl_K_4_critic',
#              ['baseline_action_repeat_4_time_ssl_K_4_critic', 'baseline_action_repeat_2_load_4_time_ssl_K_4_critic']]
# period = [0, 0, 0, 0, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_scale_load')

# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_time_ssl_time_scale_4',
#              'baseline_action_repeat_4',
#              ['baseline_action_repeat_2_time_ssl_time_scale_4', 'baseline_action_repeat_1_load_2_choose_action_bc'],
#              ['baseline_action_repeat_2_time_ssl_time_scale_4',
#               'baseline_action_repeat_1_load_2_choose_action_bc_08'],
#              ['baseline_action_repeat_4',
#               'baseline_action_repeat_2_load_4_choose_action_bc_08']
#              ]
# period = [0, 0, 0, 0, 50, 50, 50]
# plot_several_folders(prefix, folders_1, period=period, title='acrobot_swingup_same_frames_bc')

# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'baseline_action_repeat_2',
#              'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss',
#              'baseline_action_repeat_2_dyn_prior_5_previous_reverse_loss_state_20',
#              'baseline_action_repeat_2_dyn_prior_5_previous_state_20',
#              'baseline_action_repeat_2_dyn_prior_K_5_reverse_last_two']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup_dyn_prior')

# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_dynamics_reward_model_tie_dyn_critic',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_4_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_8_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_mimic_repeat_4_policy_decay_weight']
# label_list = ['drqv2', 'dyn_reward_model_tie_dyn_critic', 'action_repeat_4', 'action_repeat_8',
#               'mimic_repeat_4_policy_decay_weight']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup', label_list=label_list)

# prefix = 'reacher_hard/'
# folders_1 = ['drqv2', 'drqv2_dynamics_model', 'drqv2_dynamics_model_temporal_reflection',
#              'drqv2_dynamics_reward_model', 'drqv2_dynamics_reward_model_temporal_reflection',
#              'drqv2_dynamics_reward_model_tie_dyn_critic', 'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection']
# label_list = ['drqv2', 'dynamics_model', 'dynamics_model_temporal_reflection',
#              'dyn_reward_model', 'dyn_reward_model_temporal_reflection',
#              'dyn_reward_model_tie_dyn_critic', 'dyn_reward_model_tie_dyn_critic_temporal_reflection']
# plot_several_folders(prefix, folders_1, title='reacher_hard_architecture', label_list=label_list)
#
# folders_2 = ['drqv2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic', 'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_05',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_critic',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2']
# label_list2 = ['drqv2',
#                'dyn_reward_model_tie_dyn_critic', 'dyn_reward_model_tie_dyn_critic_time_reflect',
#                'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_05',
#                'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_2',
#                'dyn_reward_model_tie_dyn_critic_time_reflect_critic',
#                'dyn_reward_model_tie_dyn_critic_time_scale_2'
#                ]
# plot_several_folders(prefix, folders_2, title='reacher_hard_time_operation', label_list=label_list2)
#
# folders_3 = ['drqv2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_action_repeat_4_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_load_repeat_4',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_action_repeat_8_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_mimic_repeat_4_policy_decay_weight']
# label_list3 = ['drqv2',
#                'dyn_reward_model_tie_dyn_critic_time_scale_2',
#                'action_repeat_4',
#                'load_repeat_4',
#                'action_repeat_8',
#                'mimic_repeat_4_decay_weight',
#                ]
# plot_several_folders(prefix, folders_3, title='reacher_hard_action_repeat', label_list=label_list3)
#
# prefix = 'walker_run/'
# folders_2 = ['drqv2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic', 'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_05',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_critic',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_05']
# label_list2 = ['drqv2',
#                'dyn_reward_model_tie_dyn_critic', 'dyn_reward_model_tie_dyn_critic_time_reflect',
#                'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_05',
#                'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_2',
#                'dyn_reward_model_tie_dyn_critic_time_reflect_critic',
#                'dyn_reward_model_tie_dyn_critic_time_scale_05'
#                ]
# plot_several_folders(prefix, folders_2, title='walker_run_time_operation', label_list=label_list2)
#
# prefix = 'walker_run/'
# folders_3 = ['drqv2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_4_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_8_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_load_repeat_4',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_load_repeat_4_only_policy',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_mimic_repeat_4_policy',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_mimic_repeat_4_policy_decay_weight',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_1',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_1_mimic_repeat_2_policy_decay_weight']
# label_list3 = ['drqv2',
#                'dyn_reward_model_tie_dyn_critic',
#                'action_repeat_4',
#                'action_repeat_8',
#                'load_repeat_4',
#                'load_repeat_4_only_policy',
#                'mimic_repeat_4_policy',
#                'mimic_repeat_4_policy_decay_weight',
#                'repeat_1',
#                'repeat_1_mimic_repeat_2_policy_decay_weight'
#                ]
# plot_several_folders(prefix, folders_3, title='walker_run_action_repeat', label_list=label_list3)
#
# prefix = 'walker_run/'
# folders_4 = ['drqv2',
#              'drqv2_dynamics_reward_model_tie_dyn_critic',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_load_repeat_4_remove_exploration_steps',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_load_repeat_4_remove_exploration_steps_decrease_expl',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_extend_critic_repeat_4',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_extend_actor_critic_repeat_4']
# label_list4 = ['drqv2',
#                'dyn_reward_model_tie_dyn_critic',
#                'load_repeat_4_remove_expl_steps',
#                'load_repeat_4_remove_expl_steps_decrease_expl',
#                'extend_critic_repeat_4',
#                'extend_actor_critic_repeat_4'
#                ]
# plot_several_folders(prefix, folders_4, title='walker_run_load_model', label_list=label_list4)





