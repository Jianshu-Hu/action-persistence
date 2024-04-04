# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import data_augmentation
import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 2*action_shape[0]))

        self.log_std_bounds = log_std_bounds

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        mu, log_std = self.policy(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, ensemble):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.num_Qs = ensemble
        self.Q_list = nn.ModuleList()
        for i in range(self.num_Qs):
            self.Q_list.append(nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)))

        # self.Q1 = nn.Sequential(
        #     nn.Linear(feature_dim + action_shape[0], hidden_dim),
        #     nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        #
        # self.Q2 = nn.Sequential(
        #     nn.Linear(feature_dim + action_shape[0], hidden_dim),
        #     nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q_list = []
        for i in range(self.num_Qs):
            q_list.append(self.Q_list[i](h_action))
        # q_list.append(self.Q1(h_action))
        # q_list.append(self.Q2(h_action))

        return q_list


class DrQV2SACAgent:
    def __init__(self, obs_shape, action_shape, work_dir, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 aug_K, aug_type, train_dynamics_model, task_name, test_model, seed, ensemble, repeat_type):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # models
        self.obs_shape = obs_shape
        self.ensemble = ensemble
        self.encoder = Encoder(obs_shape).to(device)
        log_std_bounds = [-10, 2]
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, log_std_bounds).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.ensemble).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim, self.ensemble).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # temperature
        init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # data augmentation
        self.aug = data_augmentation.DataAug(da_type=aug_type)
        self.aug_K = aug_K

        # repeat
        self.repeat_type = repeat_type
        if repeat_type > 0:
            self.hash_count = utils.HashingBonusEvaluator(dim_key=128,
                                                          obs_processed_flat_dim=feature_dim)

        self.test_model = test_model

        self.work_dir = work_dir
        self.task_name = task_name

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs)
        if eval_mode or self.test_model:
            action = dist.mean
        else:
            action = dist.sample()
            action = action.clamp(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        target_all = []
        with torch.no_grad():
            for k in range(self.aug_K):
                dist = self.actor(next_obs[k])
                next_action = dist.rsample()
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
                target_Q_list = self.critic_target(next_obs[k], next_action)
                two_Qs_index = np.random.choice(np.arange(self.critic.num_Qs), size=2, replace=False)
                target_V = torch.min(target_Q_list[two_Qs_index[0]], target_Q_list[two_Qs_index[1]]) -\
                           self.alpha.detach() * log_prob
                target_Q = reward + (discount * target_V)
                target_all.append(target_Q)
            avg_target_Q = sum(target_all)/self.aug_K

        critic_loss_all = []
        for k in range(self.aug_K):
            Q_list = self.critic(obs[k], action)
            critic_loss = 0
            for i in range(self.critic.num_Qs):
                critic_loss += F.mse_loss(Q_list[i], avg_target_Q)
            critic_loss_all.append(critic_loss)
        avg_critic_loss = sum(critic_loss_all) / self.aug_K

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q_list[0].mean().item()
            metrics['critic_q2'] = Q_list[1].mean().item()
            metrics['critic_loss'] = avg_critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        avg_critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs[0])
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Q1, Q2 = self.critic(obs[0], action)
        # Q = torch.min(Q1, Q2)
        Q_list = self.critic(obs[0], action)
        Q = torch.min(torch.stack(Q_list), dim=0)[0]

        actor_loss = (self.alpha.detach()*log_prob-Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        # update alpha
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.use_tb:
            metrics['train_alpha_loss'] = alpha_loss.item()
            metrics['alpha_value'] = self.alpha.item()

        return metrics

    def update(self, replay_iter, step):
        if self.test_model:
            # test the model trained with larger action repeat
            obs, action, reward = next(replay_iter)
            # obs [b, t, 9, w, h]
            # action [b, t, a]
            print(obs.size())
            print(action.size())
            print(reward.size())
            idx = np.random.randint(5, obs.size(1))
            np.savez(self.work_dir+'/../../../saved_episodes/' + self.task_name + '/episodes.npz',
                     obs=obs.cpu().numpy()[:, idx-6:idx-1, :, :, :], action=action.cpu().numpy()[:, idx-5:idx, :],
                     reward=reward.cpu().numpy()[:, idx-5:idx, :])
            # obs, action, reward, discount, next_obs, one_step_next_obs, one_step_reward = \
            #     utils.to_torch(batch, self.device)
            raise ValueError('finish testing the model')
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, one_step_next_obs, one_step_reward =\
            utils.to_torch(batch, self.device)

        if self.repeat_type > 0:
            # update hash count
            with torch.no_grad():
                feature = (self.critic.trunk(self.encoder(obs.float()))).cpu().numpy()
                self.hash_count.fit_before_process_samples(feature)

        # augment
        obs_all = []
        next_obs_all = []
        for k in range(self.aug_K):
            obs_aug = self.aug(obs.float())
            next_obs_aug = self.aug(next_obs.float())
            # encoder
            obs_all.append(self.encoder(obs_aug))
            with torch.no_grad():
                next_obs_all.append(self.encoder(next_obs_aug))

        # # augment
        # obs = self.aug(obs.float())
        # next_obs = self.aug(next_obs.float())
        # # encode
        # obs = self.encoder(obs)
        # with torch.no_grad():
        #     next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs_all, action, reward, discount, next_obs_all, step))

        # update actor
        for k in range(self.aug_K):
            obs_all[k] = obs_all[k].detach()
        metrics.update(self.update_actor(obs_all, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def save(self, filename):
        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.encoder_opt.state_dict(), filename + "_encoder_optimizer")

        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_opt.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_opt.state_dict(), filename + "_actor_optimizer")

        if self.train_dynamics_model != 0:
            torch.save(self.dynamics_model.state_dict(), filename + "_dynamics_model")
            torch.save(self.dynamics_opt.state_dict(), filename + "_dynamics_optimizer")

            torch.save(self.reward_model.state_dict(), filename + "_reward_model")
            torch.save(self.reward_opt.state_dict(), filename + "_reward_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_opt.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_opt.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.encoder_opt.load_state_dict(torch.load(filename + "_encoder_optimizer"))

        if self.train_dynamics_model != 0:
            self.dynamics_model.load_state_dict(torch.load(filename + "_dynamics_model"))
            self.dynamics_opt.load_state_dict(torch.load(filename + "_dynamics_optimizer"))

            self.reward_model.load_state_dict(torch.load(filename + "_reward_model"))
            self.reward_opt.load_state_dict(torch.load(filename + "_reward_optimizer"))

    def reset(self):
        self.critic.apply(utils.weight_init)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.apply(utils.weight_init)
        if self.train_dynamics_model != 0:
            self.dynamics_model.apply(utils.weight_init)
            self.reward_model.apply(utils.weight_init)