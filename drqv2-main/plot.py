import matplotlib.pyplot as plt
import numpy as np
import math
import os

# eval_env_type = ['normal', 'color_hard', 'video_easy', 'video_hard']
eval_env_type = ['normal']


def average_over_several_runs(folder):
    mean_all = []
    std_all = []
    for env_type in range(len(eval_env_type)):
        data_all = []
        min_length = np.inf
        runs = os.listdir(folder)
        for i in range(len(runs)):
            data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
            evaluation_freq = data[2, -3]-data[1, -3]
            data_all.append(data[:, 2+env_type])
            if data.shape[0] < min_length:
                min_length = data.shape[0]
        average = np.zeros([len(runs), min_length])
        for i in range(len(runs)):
            average[i, :] = data_all[i][:min_length]
        mean = np.mean(average, axis=0)
        mean_all.append(mean)
        std = np.std(average, axis=0)
        std_all.append(std)

    return mean_all, std_all, evaluation_freq/1000


def plot_several_folders(prefix, folders, label_list=[], plot_or_save='save', title=""):
    plt.rcParams["figure.figsize"] = (6, 5)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        folder_name = 'saved_exps/'+prefix+folders[i]
        num_runs = len(os.listdir(folder_name))
        mean_all, std_all, eval_freq = average_over_several_runs(folder_name)
        for j in range(len(eval_env_type)):
            if len(eval_env_type) == 1:
                axs_plot = axs
            else:
                axs_plot = axs[int(j/2)][j-2*(int(j/2))]
            # plot variance
            axs_plot.fill_between(eval_freq*range(len(mean_all[j])),
                    mean_all[j] - std_all[j]/math.sqrt(num_runs),
                    mean_all[j] + std_all[j]/math.sqrt(num_runs), alpha=0.4)
            if len(label_list) == len(folders):
                # specify label
                axs_plot.plot(eval_freq * range(len(mean_all[j])), mean_all[j], label=label_list[i])
            else:
                axs_plot.plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=folders[i])

            axs_plot.set_xlabel('evaluation steps(x1000)')
            axs_plot.set_ylabel('episode reward')
            axs_plot.legend(fontsize=10)
            # axs_plot.set_title(eval_env_type[j])
            axs_plot.set_title(title)
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_figs/'+title)


# prefix = 'quadruped_walk/'
# # folders_1 = ['drqv2', 'drqv2_aug_2', 'drqv2_aug_2_add_KL', 'drqv2_aug_2_add_KL_add_tangent_prop']
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent_prop']
# label_list = ['drqv2', 'ours']
# plot_several_folders(prefix, folders_1, title='quadruped_walk', label_list=label_list)
#
# prefix = 'quadruped_run/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='quadruped_run', label_list=label_list)
#
# prefix = 'reach_duplo/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='reach_duplo', label_list=label_list)
#
# prefix = 'hopper_hop/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='hopper_hop', label_list=label_list)
#
#
# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup', label_list=label_list)

prefix = 'reacher_hard/'
folders_1 = ['drqv2', 'drqv2_dynamics_model', 'drqv2_dynamics_model_temporal_reflection',
             'drqv2_dynamics_reward_model', 'drqv2_dynamics_reward_model_temporal_reflection',
             'drqv2_dynamics_reward_model_tie_dyn_critic', 'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection']
label_list = ['drqv2', 'dynamics_model', 'dynamics_model_temporal_reflection',
             'dyn_reward_model', 'dyn_reward_model_temporal_reflection',
             'dyn_reward_model_tie_dyn_critic', 'dyn_reward_model_tie_dyn_critic_temporal_reflection']
plot_several_folders(prefix, folders_1, title='reacher_hard_architecture', label_list=label_list)

folders_2 = ['drqv2',
             'drqv2_dynamics_reward_model_tie_dyn_critic', 'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection',
             'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_05',
             'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_2',
             'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_critic',
             'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2']
label_list2 = ['drqv2',
               'dyn_reward_model_tie_dyn_critic', 'dyn_reward_model_tie_dyn_critic_time_reflect',
               'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_05',
               'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_2',
               'dyn_reward_model_tie_dyn_critic_time_reflect_critic',
               'dyn_reward_model_tie_dyn_critic_time_scale_2'
               ]
plot_several_folders(prefix, folders_2, title='reacher_hard_time_operation', label_list=label_list2)

folders_3 = ['drqv2',
             'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2',
             'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_action_repeat_4_save_model',
             'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_load_repeat_4',
             'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_2_action_repeat_8_save_model']
label_list3 = ['drqv2',
               'dyn_reward_model_tie_dyn_critic_time_scale_2',
               'dyn_reward_model_tie_dyn_critic_time_scale_2_action_repeat_4',
               'dyn_reward_model_tie_dyn_critic_time_scale_2_load_repeat_4',
               'dyn_reward_model_tie_dyn_critic_time_scale_2_action_repeat_8',
               ]
plot_several_folders(prefix, folders_3, title='reacher_hard_action_repeat', label_list=label_list3)

# prefix = 'finger_turn_hard/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='finger_turn_hard', label_list=label_list)

prefix = 'walker_run/'
folders_2 = ['drqv2',
             'drqv2_dynamics_reward_model_tie_dyn_critic', 'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection',
             'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_05',
             'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_time_scale_2',
             'drqv2_dynamics_reward_model_tie_dyn_critic_temporal_reflection_critic',
             'drqv2_dynamics_reward_model_tie_dyn_critic_time_scale_05']
label_list2 = ['drqv2',
               'dyn_reward_model_tie_dyn_critic', 'dyn_reward_model_tie_dyn_critic_time_reflect',
               'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_05',
               'dyn_reward_model_tie_dyn_critic_time_reflect_time_scale_2',
               'dyn_reward_model_tie_dyn_critic_time_reflect_critic',
               'dyn_reward_model_tie_dyn_critic_time_scale_05'
               ]
plot_several_folders(prefix, folders_2, title='walker_run_time_operation', label_list=label_list2)

prefix = 'walker_run/'
folders_3 = ['drqv2',
             'drqv2_dynamics_reward_model_tie_dyn_critic',
             'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_4_save_model',
             'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_8_save_model',
             'drqv2_dynamics_reward_model_tie_dyn_critic_load_repeat_4',
             'drqv2_dynamics_reward_model_tie_dyn_critic_load_repeat_4_only_policy']
label_list3 = ['drqv2',
               'dyn_reward_model_tie_dyn_critic',
               'dyn_reward_model_tie_dyn_critic_action_repeat_4',
               'dyn_reward_model_tie_dyn_critic_action_repeat_8',
               'dyn_reward_model_tie_dyn_critic_load_repeat_4',
               'dyn_reward_model_tie_dyn_critic_load_repeat_4_only_policy'
               ]
plot_several_folders(prefix, folders_3, title='walker_run_action_repeat', label_list=label_list3)

# prefix = 'finger_spin/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='finger_spin', label_list=label_list)
#
# prefix = 'cheetah_run/'
# folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
# plot_several_folders(prefix, folders_1, title='cheetah_run', label_list=label_list)




