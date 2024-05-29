from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

algorithms = ['Baseline', 'Epsilon greedy', 'Epsilon zeta', 'SNAP']
folders_all = ['drqv2', 'drqv2_epsilon_greedy', 'drqv2_epsilon_zeta', 'drqv2_batch_unvisit_repeat_nstep6_upevery4']
games = ['acrobot_swingup', 'reacher_hard', 'hopper_hop', 'walker_run', 'finger_turn_hard',
         'quadruped_run', 'humanoid_stand', 'humanoid_walk', 'humanoid_run']
normalized_score_dict = dict()
n_runs = 5
n_games = len(games)
for i in range(len(algorithms)):
    score = np.zeros((n_runs, n_games))
    for j in range(len(games)):
        dir = 'saved_exps/'+games[j]+'/'+folders_all[i]
        runs = os.listdir(dir)
        if len(runs) != n_runs:
            raise ValueError('Wrong num of runs for '+games[j]+'/'+folders_all[i])
        for k in range(len(runs)):
            data = np.loadtxt(dir + '/' + runs[k] + '/eval.csv', delimiter=',', skiprows=1)
            score[k, j] = data[-1, 2]
    normalized_score_dict[algorithms[i]] = score/1000

# plot score
# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices, each of which is of size `(num_runs x num_games)`.
# aggregate_func = lambda x: np.array([
#   metrics.aggregate_median(x),
#   metrics.aggregate_iqm(x),
#   metrics.aggregate_mean(x),
#   metrics.aggregate_optimality_gap(x)])
# aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
#     normalized_score_dict, aggregate_func, reps=50000)
# fig, axes = plot_utils.plot_interval_estimates(
#   aggregate_scores, aggregate_interval_estimates,
#   metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
#   algorithms=algorithms, xlabel_y_coordinate=-0.32, xlabel='Normalized Score')

aggregate_func = lambda x: np.array([
  metrics.aggregate_median(x),
  metrics.aggregate_mean(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_score_dict, aggregate_func, reps=50000)
fig, axes = plot_utils.plot_interval_estimates(
  aggregate_scores, aggregate_interval_estimates,
  metric_names=['Median', 'Mean'],
  algorithms=algorithms, xlabel_y_coordinate=-0.35, xlabel='Normalized Score')

file_name = '{}.png'.format('score')
fig.savefig('saved_figs/'+file_name, format='png', bbox_inches='tight')

aggregate_func = lambda x: np.array([
  metrics.aggregate_iqm(x),
  metrics.aggregate_optimality_gap(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_score_dict, aggregate_func, reps=50000)
fig, axes = plot_utils.plot_interval_estimates(
  aggregate_scores, aggregate_interval_estimates,
  metric_names=['IQM', 'Optimality Gap'],
  algorithms=algorithms, xlabel_y_coordinate=-0.35, xlabel='Normalized Score')

file_name = '{}.png'.format('score2')
fig.savefig('saved_figs/'+file_name, format='png', bbox_inches='tight')

# plot performance profile
# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices, each of which is of size `(num_runs x num_games)`.

dmc_tau = np.linspace(0.0, 1.0, 21)
perf_prof_dmc, perf_prof_dmc_cis = rly.create_performance_profile(
      normalized_score_dict, dmc_tau, reps=5000)
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(5, 3))
dmc_tau = np.linspace(0.0, 1.0, 21)
plot_utils.plot_performance_profiles(
  perf_prof_dmc, dmc_tau,
  performance_profile_cis=perf_prof_dmc_cis,
  colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
  xlabel=r'Normalized Score $(\tau)$',
  ax=ax)


ax.axhline(0.5, ls='--', color='k', alpha=0.4)
fake_patches = [mpatches.Patch(color=dict(zip(algorithms, sns.color_palette('colorblind')))[alg],
                               alpha=0.75) for alg in algorithms]
legend = fig.legend(fake_patches, algorithms, loc='upper center',
                    fancybox=True, ncol=len(algorithms),
                    # fontsize='x-large',
                    fontsize=10,
                    bbox_to_anchor=(0.5, 1.0))

file_name = '{}.png'.format('performance_profile')
fig.savefig('saved_figs/'+file_name, format='png', bbox_inches='tight')