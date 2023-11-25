import matplotlib.pyplot as plt
import numpy as np
import math
import os


def average_over_several_runs(folder):
    data_all = []
    min_length = np.inf
    runs = os.listdir(folder)
    for i in range(len(runs)):
        data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
        evaluation_freq = data[2, -3]-data[1, -3]
        data_all.append(data[:, 2])
        if data.shape[0] < min_length:
            min_length = data.shape[0]
    average = np.zeros([len(runs), min_length])
    for i in range(len(runs)):
        average[i, :] = data_all[i][:min_length]
    mean = np.mean(average, axis=0)
    std = np.std(average, axis=0)

    return mean, std, evaluation_freq/1000


def plot_several_folders(prefix, folders, period, label_list=[], plot_or_save='save', title=""):
    plt.rcParams["figure.figsize"] = (6, 5)
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
            mean_all, std_all, eval_freq = average_over_several_runs(folder_name)

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


# prefix = 'walker_run/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'baseline_action_repeat_1_load_repeat_2'],
#              'baseline_action_repeat_2_time_ssl']
# period = [0, 50, 0]
# plot_several_folders(prefix, folders_1, period=period, title='walker_run_same_frames')
#
# prefix = 'reach_duplo/'
# folders_1 = ['baseline_action_repeat_2',
#              ['baseline_action_repeat_2', 'baseline_action_repeat_1_load_repeat_2'],
#              ['baseline_action_repeat_2', 'baseline_action_repeat_1_load_repeat_2_step_125000']]
# period = [0, 50, 25]
# plot_several_folders(prefix, folders_1, period=period, title='reach_duplo_same_frames')

prefix = 'walker_run/'
folders_1 = ['baseline_action_repeat_2',
             'baseline_action_repeat_2_two_steps_invariance']
period = [0, 0]
plot_several_folders(prefix, folders_1, period=period, title='walker_run_two_steps_invariance')

prefix = 'walker_run/'
folders_1 = ['baseline_action_repeat_2',
             'baseline_action_repeat_2_time_ssl',
             'baseline_action_repeat_2_time_ssl_perturb_3', 'baseline_action_repeat_2_time_ssl_perturb_4',
             'baseline_action_repeat_2_time_ssl_perturb_3_trainable_temp']
period = [0, 0]
plot_several_folders(prefix, folders_1, period=period, title='walker_run_ssl')

prefix = 'reacher_hard/'
folders_1 = ['baseline_action_repeat_2',
             'baseline_action_repeat_2_two_steps_invariance',
             'baseline_action_repeat_2_two_steps_invariance_weight_1']
period = [0, 0]
plot_several_folders(prefix, folders_1, period=period, title='reacher_hard_two_steps_invariance')

# prefix = 'acrobot_swingup/'
# folders_1 = ['drqv2', 'drqv2_dynamics_reward_model_tie_dyn_critic',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_4_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_action_repeat_8_save_model',
#              'drqv2_dynamics_reward_model_tie_dyn_critic_mimic_repeat_4_policy_decay_weight']
# label_list = ['drqv2', 'dyn_reward_model_tie_dyn_critic', 'action_repeat_4', 'action_repeat_8',
#               'mimic_repeat_4_policy_decay_weight']
# plot_several_folders(prefix, folders_1, title='acrobot_swingup', label_list=label_list)
#
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





