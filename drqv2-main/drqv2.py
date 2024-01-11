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
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward_mu_std(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return mu, std

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class ExActor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.loaded_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.trained_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.copied_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def load(self, old_actor):
        # load the old actor
        self.trunk.load_state_dict(old_actor.trunk.state_dict())
        self.loaded_policy.load_state_dict(old_actor.policy.state_dict())

        for param in self.loaded_policy.parameters():
            param.requires_grad = False

        # copy policy
        self.copied_policy.load_state_dict(self.trained_policy.state_dict())

        for param in self.copied_policy.parameters():
            param.requires_grad = False

    def forward_mu_std(self, obs, std):
        h = self.trunk(obs)

        loaded_mu = torch.tanh(self.loaded_policy(h))
        std = torch.ones_like(loaded_mu) * std

        trained_mu = torch.tanh(self.trained_policy(h))

        copied_mu = torch.tanh(self.copied_policy(h))
        return loaded_mu+trained_mu-copied_mu, std

    def forward(self, obs, std):
        final_mu, final_std = self.forward_mu_std(obs, std)

        dist = utils.TruncatedNormal(final_mu, final_std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.num_Qs = 2
        self.Q_list = nn.ModuleList()
        for i in range(self.num_Qs):
            self.Q_list.append(nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)))

        # self.Q2 = nn.Sequential(
        #     nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q_list = []
        for i in range(self.num_Qs):
            q_list.append(self.Q_list[i](h_action))

        return q_list


class ExCritic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.loaded_Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.loaded_Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.trained_Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.trained_Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.copied_Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.copied_Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def load(self, old_critic):
        # load the critic
        self.trunk.load_state_dict(old_critic.trunk.state_dict())
        self.loaded_Q1.load_state_dict(old_critic.Q1.state_dict())
        self.loaded_Q2.load_state_dict(old_critic.Q2.state_dict())

        for param in self.loaded_Q1.parameters():
            param.requires_grad = False
        for param in self.loaded_Q2.parameters():
            param.requires_grad = False

        # copy critic
        self.copied_Q1.load_state_dict(self.trained_Q1.state_dict())
        self.copied_Q2.load_state_dict(self.trained_Q2.state_dict())

        for param in self.copied_Q1.parameters():
            param.requires_grad = False
        for param in self.copied_Q2.parameters():
            param.requires_grad = False

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        loaded_q1 = self.loaded_Q1(h_action)
        loaded_q2 = self.loaded_Q2(h_action)

        copied_q1 = self.copied_Q1(h_action)
        copied_q2 = self.copied_Q2(h_action)

        trained_q1 = self.trained_Q1(h_action)
        trained_q2 = self.trained_Q2(h_action)

        return loaded_q1+trained_q1-copied_q1, loaded_q2+trained_q2-copied_q2


class RewardModel(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.reward_model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, h):
        r = self.reward_model(h)

        return r


class DynamicsModel(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, critic_trunk):
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())
        self.trunk = critic_trunk

        self.dynamics_model = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        next_state = self.dynamics_model(h_action)

        return next_state

    # def forward_two_steps(self, obs, action_1, action_2):
    #     h = self.trunk(obs)
    #     h_action = torch.cat([h, action_1], dim=-1)
    #     next_state = self.dynamics_model(h_action)
    #
    #     h_action_next = torch.cat([next_state, action_2], dim=-1)
    #     next_next_state = self.dynamics_model(h_action_next)
    #
    #     return next_next_state


class InvDynamicsModel(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, critic_trunk):
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())
        self.trunk = critic_trunk

        self.inv_dynamics_model = nn.Sequential(
            nn.Linear(2*feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))

        self.predict_action_model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward_delta_s(self, obs, next_obs):
        h_0 = self.trunk(obs)
        h_1 = self.trunk(next_obs)
        h = torch.cat([h_0, h_1], dim=-1)
        delta_s = self.inv_dynamics_model(h)
        return delta_s

    def forward(self, obs, next_obs):
        delta_s = self.forward_delta_s(obs,next_obs)
        predict_a = self.predict_action_model(delta_s)

        return predict_a


class DisentangledDynamicsModel(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, critic_trunk):
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())
        self.trunk = critic_trunk

        self.linear_dynamics_model = nn.Sequential(
            nn.Linear(feature_dim+action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))

        self.nonlinear_dynamics_model = nn.Sequential(
            nn.Linear(feature_dim+action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        delta_s_linear = self.linear_dynamics_model(h_action)
        delta_s_nonlinear = self.nonlinear_dynamics_model(h_action)

        return delta_s_linear, delta_s_nonlinear


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, work_dir, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 aug_K, aug_type, add_KL_loss, tangent_prop, train_dynamics_model,
                 load_model, load_folder, pretrain_steps, task_name, test_model, seed,
                 time_ssl_K, time_ssl_weight, dyn_prior_K, dyn_prior_weight, state_dim):
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
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = data_augmentation.DataAug(da_type=aug_type)
        self.aug_K = aug_K

        # KL regularization in actor training
        self.add_KL_loss = add_KL_loss
        # tangent prop regularization
        self.tangent_prop = tangent_prop

        # train a dynamics model
        self.train_dynamics_model = train_dynamics_model
        if self.train_dynamics_model == 1:
            self.dynamics_model = DynamicsModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.critic.trunk).to(device)
            self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)

            self.reward_model = RewardModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)
        elif self.train_dynamics_model == 2:
            self.dynamics_model = InvDynamicsModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.critic.trunk).to(device)
            self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)

            self.reward_model = RewardModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)
        elif self.train_dynamics_model == 3:
            self.dynamics_model = DisentangledDynamicsModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.critic.trunk).to(device)
            self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)

            self.reward_model = RewardModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)


        self.encoder_mimic = Encoder(obs_shape).to(device)
        self.actor_mimic = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        # temporal self-supervised loss
        self.time_ssl_K = time_ssl_K
        self.time_ssl_weight = time_ssl_weight
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-06)
        self.W = nn.Parameter(torch.ones(1).to(device)/5)
        self.W_opt = torch.optim.Adam([{'params': self.W, 'lr': lr}])
        # dynamics prior knowledge
        self.dyn_prior_K = dyn_prior_K
        self.dyn_prior_weight = dyn_prior_weight
        self.state_dim = state_dim

        # load model
        self.work_dir = work_dir
        self.load_model = load_model
        self.load_folder = load_folder
        self.pretrain_steps = pretrain_steps
        self.test_model = test_model
        self.task_name = task_name
        if load_model != 'none':
            # self.extend_model = False
            # if self.extend_model:
            #     self.critic = ExCritic(self.encoder.repr_dim, action_shape, feature_dim,
            #                            hidden_dim).to(device)
            #     self.critic_target = ExCritic(self.encoder.repr_dim, action_shape,
            #                                   feature_dim, hidden_dim).to(device)
            #     # optimizers
            #     self.critic_opt = torch.optim.Adam(list(self.critic.trunk.parameters()) +
            #                                        list(self.critic.trained_Q1.parameters()) + list(
            #         self.critic.trained_Q2.parameters()), lr=lr)
            #
            #     self.actor = ExActor(self.encoder.repr_dim, action_shape, feature_dim,
            #                            hidden_dim).to(device)
            #     # optimizers
            #     self.actor_opt = torch.optim.Adam(list(self.actor.trunk.parameters()) +
            #                                        list(self.actor.trained_policy.parameters()), lr=lr)

            self.load(self.work_dir+'/../../../saved_model/' + task_name + '/'+str(self.load_folder)
                      + '/' + 'seed_' + str(seed)+'/' + load_model)
            print('load model from: ')
            print(self.work_dir+'/../../../saved_model/' + task_name + '/'+str(self.load_folder)
                  + '/' + 'seed_'+str(seed)+'/' + load_model)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        if self.train_dynamics_model != 0:
            self.dynamics_model.train(training)
            self.reward_model.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode or self.test_model:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps and self.load_model == 'none':
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def loaded_policy_act(self, last_2_obs, last_1_obs, obs, step, eval_mode):
        last_2_obs = torch.as_tensor(last_2_obs, device=self.device)
        last_1_obs = torch.as_tensor(last_1_obs, device=self.device)
        obs = torch.as_tensor(obs, device=self.device)
        scaled_obs = torch.clone(obs)
        scaled_obs[0:3, :, :] = last_2_obs[0:3, :, :]
        scaled_obs[3:6, :, :] = obs[0:3, :, :]

        scaled_obs = self.encoder_mimic(scaled_obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor_mimic(scaled_obs, stddev)
        if eval_mode or self.test_model:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def tangent_vector(self, obs):
        pad = nn.Sequential(torch.nn.ReplicationPad2d(1))
        pad_obs = pad(obs)
        index = np.random.randint(4, size=1)[0]
        if index == 0:
            # horizontal shift 1 pixel
            obs_aug = torchvision.transforms.functional.crop(pad_obs, top=1, left=2, height=obs.shape[-1], width=obs.shape[-1])
        elif index == 1:
            # horizontal shift 1 pixel
            obs_aug = torchvision.transforms.functional.crop(pad_obs, top=1, left=0, height=obs.shape[-1], width=obs.shape[-1])
        elif index == 2:
            # vertical shift 1 pixel
            obs_aug = torchvision.transforms.functional.crop(pad_obs, top=2, left=1, height=obs.shape[-1], width=obs.shape[-1])
        elif index == 3:
            # vertical shift 1 pixel
            obs_aug = torchvision.transforms.functional.crop(pad_obs, top=0, left=1, height=obs.shape[-1], width=obs.shape[-1])
        tan_vector = obs_aug - obs
        return tan_vector

    def update_critic(self, obs, action, reward, discount, next_obs, step, obs_original, next_K_step_obs):
        metrics = dict()

        target_all = []
        with torch.no_grad():
            for k in range(self.aug_K):
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs[k], stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Q_list = self.critic_target(next_obs[k], next_action)
                two_Qs_index = np.random.choice(np.arange(self.critic.num_Qs), size=2, replace=False)
                target_V = torch.min(target_Q_list[two_Qs_index[0]], target_Q_list[two_Qs_index[1]])
                target_Q = reward + (discount * target_V)
                target_all.append(target_Q)
            avg_target_Q = sum(target_all)/self.aug_K

        if self.tangent_prop:
            obs_aug_1 = self.aug(obs_original)
            obs_aug_2 = self.aug(obs_original)
            with torch.no_grad():
                # calculate the tangent vector
                tangent_vector1 = self.tangent_vector(obs_aug_1)
                tangent_vector2 = self.tangent_vector(obs_aug_2)
            obs_aug_1.requires_grad = True
            obs_aug_2.requires_grad = True
            # critic loss
            Q1_aug_1, Q2_aug_1 = self.critic(self.encoder(obs_aug_1), action)
            Q1_aug_2, Q2_aug_2 = self.critic(self.encoder(obs_aug_2), action)
            critic_loss = F.mse_loss(Q1_aug_1, avg_target_Q) + F.mse_loss(Q2_aug_1, avg_target_Q)
            critic_loss += F.mse_loss(Q1_aug_2, avg_target_Q) + F.mse_loss(Q2_aug_2, avg_target_Q)
            avg_critic_loss = critic_loss / 2

            # add regularization for tangent prop
            # calculate the Jacobian matrix for non-linear model
            Q1 = torch.min(Q1_aug_1, Q2_aug_1)
            jacobian1 = torch.autograd.grad(outputs=Q1, inputs=obs_aug_1,
                                           grad_outputs=torch.ones(Q1.size(), device=self.device),
                                           retain_graph=True, create_graph=True)[0]
            Q2 = torch.min(Q1_aug_2, Q2_aug_2)
            jacobian2 = torch.autograd.grad(outputs=Q2, inputs=obs_aug_2,
                                           grad_outputs=torch.ones(Q2.size(), device=self.device),
                                           retain_graph=True, create_graph=True)[0]
            tan_loss1 = torch.mean(torch.square(torch.sum((jacobian1 * tangent_vector1), (3, 2, 1))), dim=-1)
            tan_loss2 = torch.mean(torch.square(torch.sum((jacobian2 * tangent_vector2), (3, 2, 1))), dim=-1)

            tangent_prop_loss = tan_loss1+tan_loss2
            avg_critic_loss += 0.1*tangent_prop_loss

            if self.use_tb:
                metrics['tangent_prop_loss'] = tangent_prop_loss.item()
        else:
            critic_loss_all = []
            for k in range(self.aug_K):
                Q_list = self.critic(obs[k], action)
                critic_loss = 0
                for i in range(self.critic.num_Qs):
                    critic_loss += F.mse_loss(Q_list[i], avg_target_Q)
                critic_loss_all.append(critic_loss)
            avg_critic_loss = sum(critic_loss_all) / self.aug_K

        if self.dyn_prior_K > 0:
            # disentangle the features
            with torch.no_grad():
                target_obs = torch.clone(next_K_step_obs[:, self.dyn_prior_K-1, :, :, :])
                target_feature = self.critic.trunk(self.encoder(target_obs))
            # position information is invariant with respect to the change of first two frames
            second_frame_index = np.random.randint(0, self.dyn_prior_K)
            first_frame_index = np.random.randint(second_frame_index, self.dyn_prior_K)
            pos_inv_obs = torch.clone(next_K_step_obs[:, self.dyn_prior_K-1, :, :, :])
            pos_inv_obs[:, 0:3, :, :] = next_K_step_obs[:, self.dyn_prior_K - 1 - first_frame_index, 0:3, :, :]
            pos_inv_obs[:, 3:6, :, :] = next_K_step_obs[:, self.dyn_prior_K - 1 - second_frame_index, 3:6, :, :]
            pos_inv_feature = self.critic.trunk(self.encoder(pos_inv_obs))
            mse_1 = torch.nn.functional.mse_loss(pos_inv_feature[:, 0:self.state_dim], target_feature[:, 0:self.state_dim])

            # velocity information is invariant with respect to the change of first frames
            first_frame_index = np.random.randint(0, self.dyn_prior_K)
            vel_inv_obs = torch.clone(next_K_step_obs[:, self.dyn_prior_K-1, :, :, :])
            vel_inv_obs[:, 0:3, :, :] = next_K_step_obs[:, self.dyn_prior_K-1 - first_frame_index, 0:3, :, :]
            vel_inv_feature = self.critic.trunk(self.encoder(vel_inv_obs))
            mse_2 = torch.nn.functional.mse_loss(vel_inv_feature[:, self.state_dim:2*self.state_dim],
                                     target_feature[:, self.state_dim:2*self.state_dim])

            # dyn_prior_loss = -torch.mean(cos_sim_1+cos_sim_2)
            dyn_prior_loss = mse_1+mse_2
            avg_critic_loss += self.dyn_prior_weight*dyn_prior_loss

            # reverse loss
            with torch.no_grad():
                normal_obs_feature = self.critic.trunk(self.encoder(next_K_step_obs[:, 0, :, :, :]))
                reflected_obs = self.time_reflect_obs(next_K_step_obs[:, 1, :, :, :])
            reflected_feature = self.critic.trunk(self.encoder(reflected_obs))
            # position information is invariant with respect to the time reversal
            # cos_sim_3 = self.cos_sim(reflected_feature[:, 0:self.state_dim], normal_obs_feature[:, 0:self.state_dim])

            # velocity information is equivariant with respect to the time reversal
            mse_4 = torch.nn.functional.mse_loss(reflected_feature[:, self.state_dim:2*self.state_dim],
                                     -normal_obs_feature[:, self.state_dim:2*self.state_dim])

            # temporal_reverse_loss = -torch.mean(cos_sim_3+cos_sim_4)
            temporal_reverse_loss = mse_4
            avg_critic_loss += self.dyn_prior_weight*temporal_reverse_loss

        if self.time_ssl_K > 0:
            # previous states
            # s_t = (nt-2n,nt-n,nt)
            # s_{t-1} = (nt-3n,nt-2n,nt-n)
            # s_{t-2} = (nt-4n,nt-3n,nt-2n)
            # ...
            # s_{t-k} = (nt-(k+2)n,nt-(k+1)n,nt-kn)
            with torch.no_grad():
                target_obs = torch.clone(next_K_step_obs[:, -1, :, :, :])
                target_obs_feature = self.critic.trunk(self.encoder(target_obs))
            sim_list = torch.zeros([target_obs_feature.size(0), self.time_ssl_K-1]).to(target_obs_feature.device)
            # calculate the cosine similarities for scaled obs
            for scale in range(1, self.time_ssl_K):
                obs_scale = torch.clone(next_K_step_obs[:, -1, :, :, :])
                obs_scale[:, 0:3, :, :] = next_K_step_obs[:, -(scale*2+1), 0:3, :, :]
                obs_scale[:, 3:6, :, :] = next_K_step_obs[:, -scale, 0:3, :, :]
                pert_obs_feature = self.critic.trunk(self.encoder(obs_scale))
                sim = self.cos_sim(pert_obs_feature, target_obs_feature)
                sim_list[:, scale-1] = torch.exp(sim*self.W)
            # # cross entropy loss
            for sim_i in range(self.time_ssl_K-2):
                ssloss = torch.mean(-torch.log(sim_list[:, sim_i]/(sim_list[:, sim_i]+sim_list[:, sim_i+1])))
                critic_loss += self.time_ssl_weight*ssloss
            # Rank-N-Contrast
            # for sim_i in range(self.time_ssl_K-2):
            #     ssloss = torch.mean(-torch.log(sim_list[:, sim_i]/torch.sum(sim_list[:, sim_i:], dim=-1)))
            #     critic_loss += self.time_ssl_weight*ssloss

        # optimize W
        # if self.time_ssl_K > 0:
        #     self.W_opt.zero_grad(set_to_none=True)
        #     self.W_opt.step()
        #     if step % 1000 == 0:
        #         print('W: ')
        #         print(self.W)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = avg_critic_loss.item()

            # if self.time_ssl_K > 0:
            #     metrics['ss_loss'] = ssloss.item()
            if self.dyn_prior_K > 0:
                metrics['dyn_prior_loss'] = dyn_prior_loss.item()
                metrics['temporal_reverse_loss'] = temporal_reverse_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        avg_critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs[0], stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Q1, Q2 = self.critic(obs[0], action)
        # Q = torch.min(Q1, Q2)
        Q_list = self.critic(obs[0], action)
        Q = sum(Q_list)/len(Q_list)

        actor_loss = -Q.mean()

        if self.add_KL_loss:
            # KL divergence between A(obs_aug_1) and A(obs_aug_2)
            with torch.no_grad():
                mu_aug_1, std_aug_1 = self.actor.forward_mu_std(obs[0], stddev)
            mu_aug_2, std_aug_2 = self.actor.forward_mu_std(obs[1], stddev)
            dist_aug_1 = torch.distributions.Normal(mu_aug_1, std_aug_1)
            dist_aug_2 = torch.distributions.Normal(mu_aug_2, std_aug_2)

            KL = torch.mean(torch.distributions.kl_divergence(dist_aug_1, dist_aug_2))
            weighted_KL = 0.1 * KL
            actor_loss += weighted_KL

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.add_KL_loss:
                metrics['actor_KL_loss'] = KL.item()

            # if self.time_ssl_K > 0:
            #     metrics['ss_loss'] = ssloss.item()

        return metrics

    def pretrain(self, replay_iter):
        # train encoder and critic first
        for step in range(self.pretrain_steps):
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, one_step_next_obs, one_step_reward, next_K_step_obs, t_index = \
                utils.to_torch(batch, self.device)
            # aug
            obs_all = []
            next_obs_all = []
            for k in range(self.aug_K):
                obs_aug = self.aug(obs.float())
                next_obs_aug = self.aug(next_obs.float())
                # encoder
                obs_all.append(self.encoder(obs_aug))
                with torch.no_grad():
                    next_obs_all.append(self.encoder(next_obs_aug))

            # update critic using normal critic loss
            self.update_critic(obs_all, action, reward, discount, next_obs_all, step, obs.float(), next_K_step_obs)

            # update dynamics and reward model
            with torch.no_grad():
                one_step_next_obs_aug = self.aug(one_step_next_obs.float())
            if self.train_dynamics_model != 0:
                self.update_dynamics_reward_model(obs_aug, action, one_step_reward, one_step_next_obs_aug, next_K_step_obs)

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        # train actor
        for step in range(self.pretrain_steps):
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, one_step_next_obs, one_step_reward, next_K_step_obs, t_index = \
                utils.to_torch(batch, self.device)
            # aug
            with torch.no_grad():
                obs_aug = self.aug(obs.float())
                obs_aug = self.encoder(obs_aug)

            # update actor with a behavior cloning loss
            mu, std = self.actor.forward_mu_std(obs_aug, std=0)
            bc_loss = torch.nn.functional.mse_loss(mu, action)
            weighted_bc_loss = 1.0*bc_loss
            if step % 1000 == 0:
                print('pretraining actor step: ' + str(step) + ' loss: ' + str(weighted_bc_loss.item()))
            self.actor_opt.zero_grad(set_to_none=True)
            weighted_bc_loss.backward()
            self.actor_opt.step()

        # # extend the critic
        # old_critic = copy.deepcopy(self.critic)
        # self.critic = ExCritic(self.encoder.repr_dim, self.action_shape,
        #                        self.feature_dim, self.hidden_dim).to(self.device)
        # self.critic.load(old_critic)
        # del old_critic
        # self.critic_target = ExCritic(self.encoder.repr_dim, self.action_shape,
        #                               self.feature_dim, self.hidden_dim).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        #
        # # optimizers
        # self.critic_opt = torch.optim.Adam(list(self.critic.trunk.parameters()) +
        #                                    list(self.critic.trained_Q1.parameters()) +
        #                                    list(self.critic.trained_Q2.parameters()), lr=self.lr)
        # # extend actor
        # old_actor = copy.deepcopy(self.actor)
        # self.actor = ExActor(self.encoder.repr_dim, self.action_shape, self.feature_dim,
        #                        self.hidden_dim).to(self.device)
        # self.actor.load(old_actor)
        # del old_actor
        # # optimizers
        # self.actor_opt = torch.optim.Adam(list(self.actor.trunk.parameters()) +
        #                                    list(self.actor.trained_policy.parameters()), lr=self.lr)

    def time_reflect_obs(self, obs):
        stacked_num = int(obs.size(1)/3)
        reflected_obs = torch.zeros_like(obs).to(obs.device)
        for i in range(stacked_num):
            reflected_obs[:, 3 * i:3 * (i + 1), :, :] = obs[:, obs.size(1)-3 * (i + 1):obs.size(1)-3 * i, :, :]
        return reflected_obs

    def update_dynamics_reward_model(self, obs, action, reward, next_obs, next_K_obs):
        metrics = dict()
        if self.train_dynamics_model == 1:
            predicted_next_obs_feature = self.dynamics_model(self.encoder(obs), action)
            with torch.no_grad():
                next_obs_feature = self.dynamics_model.trunk(self.encoder(next_obs))
            dynamics_loss = torch.nn.functional.mse_loss(predicted_next_obs_feature, next_obs_feature)

            predicted_reward = self.reward_model(predicted_next_obs_feature)
            reward_loss = torch.nn.functional.mse_loss(predicted_reward, reward)
        elif self.train_dynamics_model == 2:
            delta_s = self.dynamics_model.forward_delta_s(self.encoder(obs), self.encoder(next_obs))
            predicted_action = self.dynamics_model.predict_action_model(delta_s)
            dynamics_loss = torch.nn.functional.mse_loss(predicted_action, action)

            predicted_reward = self.reward_model(self.dynamics_model.trunk(self.encoder(next_obs)))
            reward_loss = torch.nn.functional.mse_loss(predicted_reward, reward)

            # time reverse loss
            original_delta_s = self.dynamics_model.forward_delta_s(self.encoder(next_K_obs[:, 0, :, :, :]),
                                                                   self.encoder(next_K_obs[:, 1, :, :, :]))
            reflected_obs = self.time_reflect_obs(next_K_obs[:, 2, :, :, :])
            reflected_next_obs = self.time_reflect_obs(next_K_obs[:, 3, :, :, :])
            reverse_delta_s = self.dynamics_model.forward_delta_s(self.encoder(reflected_obs),
                                                                  self.encoder(reflected_next_obs))
            time_reverse_loss = torch.nn.functional.mse_loss(original_delta_s, -reverse_delta_s)
            # reverse_delta_s = self.dynamics_model.forward_delta_s(self.encoder(next_obs),
            #                                                       self.encoder(obs))
            # time_reverse_loss = torch.nn.functional.mse_loss(delta_s, -reverse_delta_s)
            dynamics_loss += time_reverse_loss
        elif self.train_dynamics_model == 3:
            encoded_obs = self.encoder(obs)
            obs_feature = self.dynamics_model.trunk(encoded_obs)
            linear_delta_s, nonlinear_delta_s = self.dynamics_model(encoded_obs, action)
            predicted_next_obs_feature = obs_feature+linear_delta_s+nonlinear_delta_s
            with torch.no_grad():
                next_obs_feature = self.dynamics_model.trunk(self.encoder(next_obs))
            dynamics_loss = torch.nn.functional.mse_loss(predicted_next_obs_feature, next_obs_feature)

            predicted_reward = self.reward_model(predicted_next_obs_feature)
            reward_loss = torch.nn.functional.mse_loss(predicted_reward, reward)

            # linear loss
            scale_1 = (0.5+torch.rand(action.size(0), 1).to(action.device))
            scale_2 = (0.5 + torch.rand(action.size(0), 1).to(action.device))
            scale_1_delta_s, nonlinear_scale1_delta_s = self.dynamics_model(encoded_obs, scale_1 * action)
            scale_2_delta_s, nonlinear_scale2_delta_s = self.dynamics_model(encoded_obs, scale_2 * action)
            linear_loss = torch.nn.functional.mse_loss((linear_delta_s.detach()-scale_1_delta_s)*(1-scale_2),
                                                       (linear_delta_s.detach()-scale_2_delta_s)*(1-scale_1))
            dynamics_loss += linear_loss

            # non-linear loss
            l1loss = torch.nn.L1Loss()
            non_linear_loss = l1loss(nonlinear_delta_s,
                                              torch.zeros_like(nonlinear_delta_s).to(nonlinear_delta_s.device))
            dynamics_loss += non_linear_loss


        loss = reward_loss+dynamics_loss
        # optimize dynamics model, reward model and encoder
        self.encoder_opt.zero_grad(set_to_none=True)
        self.dynamics_opt.zero_grad(set_to_none=True)
        self.reward_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_opt.step()
        self.dynamics_opt.step()
        self.reward_opt.step()

        if self.use_tb:
            metrics['dynamics_loss'] = dynamics_loss.item()
            metrics['reward_loss'] = reward_loss.item()

        return metrics

    def update(self, replay_iter, step, old_replay_iter=None):
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

        if old_replay_iter is None:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, one_step_next_obs, one_step_reward, next_K_step_obs, t_index = \
                utils.to_torch(batch, self.device)
        else:
            old_batch = next(old_replay_iter)
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, one_step_next_obs, one_step_reward, next_K_step_obs, t_index = \
                utils.two_batches_to_torch(batch, old_batch, self.device)

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

        # next_K_step_obs: [b,K,9,w,h]
        next_K_step_obs = next_K_step_obs.float()
        for k in range(next_K_step_obs.size(1)):
            next_K_step_obs[:, k, :, :, :] = self.aug(next_K_step_obs[:, k, :, :, :])

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
            self.update_critic(obs_all, action, reward, discount, next_obs_all, step,
                               obs.float(), next_K_step_obs))

        # update dynamics and reward model
        with torch.no_grad():
            one_step_next_obs_aug = self.aug(one_step_next_obs.float())
        if self.train_dynamics_model != 0:
            metrics.update(self.update_dynamics_reward_model(obs_aug, action, one_step_reward,
                                                             one_step_next_obs_aug, next_K_step_obs))

        # update actor
        for k in range(self.aug_K):
            obs_all[k] = obs_all[k].detach()
        metrics.update(self.update_actor(obs_all, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

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
        # if self.extend_model:
        #     self.critic.load(filename)
        #     self.critic_target.load_state_dict(self.critic.state_dict())
        #
        #     self.actor.load(filename)
        # else:

        self.actor_mimic.load_state_dict(torch.load(filename + "_actor"))
        self.encoder_mimic.load_state_dict(torch.load(filename + "_encoder"))

        # self.critic.load_state_dict(torch.load(filename + "_critic"))
        # # self.critic_opt.load_state_dict(torch.load(filename + "_critic_optimizer"))
        # self.critic_target.load_state_dict(self.critic.state_dict())
        #
        # self.actor.load_state_dict(torch.load(filename + "_actor"))
        # # self.actor_opt.load_state_dict(torch.load(filename + "_actor_optimizer"))
        #
        # self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        # # self.encoder_opt.load_state_dict(torch.load(filename + "_encoder_optimizer"))
        #
        # if self.train_dynamics_model != 0:
        #     self.dynamics_model.load_state_dict(torch.load(filename + "_dynamics_model"))
        #     # self.dynamics_opt.load_state_dict(torch.load(filename + "_dynamics_optimizer"))
        #
        #     self.reward_model.load_state_dict(torch.load(filename + "_reward_model"))
        #     # self.reward_opt.load_state_dict(torch.load(filename + "_reward_optimizer"))