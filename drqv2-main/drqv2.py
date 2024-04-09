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

from sklearn.cluster import KMeans


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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, device, max_len=500):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.max_len = max_len
        self.position = torch.arange(0, max_len).unsqueeze(1).to(device)
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).to(device)
        self.div_term.requires_grad = True

    def forward(self, position):
        position = position.reshape(-1,)
        pe = torch.zeros(self.max_len, self.d_model).to(position.device)
        pe[:, 0::2] = torch.sin(self.position * self.div_term)
        pe[:, 1::2] = torch.cos(self.position * self.div_term)
        x = pe[position, :]
        return x


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, device, noisy_net=False, pos_emb=False,
                 pos_emb_dim=0):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        if noisy_net:
            linear = utils.NoisyLinear
        else:
            linear = nn.Linear
        self.linear = [linear(hidden_dim, hidden_dim), linear(hidden_dim, action_shape[0])]
        self.pos_emb = pos_emb
        self.pos_emb_dim = pos_emb_dim
        if pos_emb:
            layers = [nn.Linear(feature_dim+pos_emb_dim, hidden_dim),
                      nn.ReLU(inplace=True), self.linear[0],
                      nn.ReLU(inplace=True), self.linear[1]]
        else:
            layers = [nn.Linear(feature_dim, hidden_dim),
                      nn.ReLU(inplace=True), self.linear[0],
                      nn.ReLU(inplace=True), self.linear[1]]
        # self.linear = [linear(feature_dim, hidden_dim),
        #                linear(hidden_dim, hidden_dim), linear(hidden_dim, action_shape[0])]
        # layers = [self.linear[0],
        #           nn.ReLU(inplace=True), self.linear[1],
        #           nn.ReLU(inplace=True), self.linear[2]]
        self.policy = nn.Sequential(*layers)

        # self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(hidden_dim, hidden_dim),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)
        if noisy_net:
            for module in self.linear:
                module.reset_parameters()

    def reset_noise(self):
        for module in self.linear:
            module.reset_noise()

    def forward_mu_std(self, obs, std, pos=None):
        h = self.trunk(obs)
        if self.pos_emb:
            h = torch.cat((h, pos), dim=1)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return mu, std

    def forward(self, obs, std, pos=None):
        h = self.trunk(obs)

        if self.pos_emb:
            h = torch.cat((h, pos), dim=1)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, device, ensemble, noisy_net=False,
                 pos_emb=False, pos_emb_dim=0):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.pos_emb = pos_emb
        self.pos_emb_dim = pos_emb_dim

        self.num_Qs = ensemble
        self.Q_list = nn.ModuleList()
        self.linear_list = []
        if noisy_net:
            linear = utils.NoisyLinear
        else:
            linear = nn.Linear
        for i in range(self.num_Qs):
            self.linear_list.append([linear(hidden_dim, hidden_dim),
                                    linear(hidden_dim, 1)])
            if self.pos_emb:
                layers = [nn.Linear(feature_dim + action_shape[0] + self.pos_emb_dim, hidden_dim),
                    nn.ReLU(inplace=True), self.linear_list[i][0],
                    nn.ReLU(inplace=True), self.linear_list[i][1]]
            else:
                layers = [nn.Linear(feature_dim + action_shape[0], hidden_dim),
                    nn.ReLU(inplace=True), self.linear_list[i][0],
                    nn.ReLU(inplace=True), self.linear_list[i][1]]
            # self.linear_list.append([linear(feature_dim + action_shape[0], hidden_dim),
            #                         linear(hidden_dim, hidden_dim),
            #                         linear(hidden_dim, 1)])
            # layers = [self.linear_list[i][0],
            #     nn.ReLU(inplace=True), self.linear_list[i][1],
            #     nn.ReLU(inplace=True), self.linear_list[i][2]]
            self.Q_list.append(nn.Sequential(*layers))

        self.apply(utils.weight_init)
        if noisy_net:
            for i in range(self.num_Qs):
                for module in self.linear_list[i]:
                    module.reset_parameters()

    def reset_noise(self):
        for i in range(self.num_Qs):
            for module in self.linear_list[i]:
                module.reset_noise()

    def forward(self, obs, action, pos=None):
        h = self.trunk(obs)

        if self.pos_emb:
            h = torch.cat((h, pos), dim=1)

        h_action = torch.cat([h, action], dim=-1)
        q_list = []
        for i in range(self.num_Qs):
            q_list.append(self.Q_list[i](h_action))
        # q_list.append(self.Q1(h_action))
        # q_list.append(self.Q2(h_action))

        return q_list


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


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, work_dir, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 aug_K, aug_type, train_dynamics_model, task_name, test_model, seed, ensemble, repeat_type,
                 repeat_coefficient,
                 epsilon_greedy, epsilon_schedule, epsilon_zeta, noisy_net, load_folder, load_model,
                 temp_cluster, pos_emb):
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
        self.noisy_net = noisy_net

        self.pos_emb = pos_emb
        self.pos_emb_dim = 10
        self.pe = PositionalEmbedding(d_model=self.pos_emb_dim, device=device)

        # models
        self.obs_shape = obs_shape
        self.ensemble = ensemble
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, device, self.noisy_net, self.pos_emb, self.pos_emb_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, device, self.ensemble, self.noisy_net, self.pos_emb,
                             self.pos_emb_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim, device, self.ensemble, self.noisy_net, self.pos_emb,
                                    self.pos_emb_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.pos_emb_opt = torch.optim.Adam([self.pe.div_term], lr=lr)

        # data augmentation
        self.aug = data_augmentation.DataAug(da_type=aug_type)
        self.aug_K = aug_K

        # train a dynamics model
        self.train_dynamics_model = train_dynamics_model
        if self.train_dynamics_model == 1:
            self.dynamics_model = DynamicsModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.critic.trunk).to(device)
            self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)

            self.reward_model = RewardModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

        # repeat
        self.repeat_type = repeat_type
        self.load_folder = load_folder
        if load_folder != 'None':
            filename = work_dir+'/../../../saved_model/manipulation_'+task_name+'/'+load_folder+'/seed_'+str(seed)+'/'+\
                       load_model
            print(filename)
            self.encoder_repeat = Encoder(obs_shape).to(device)
            self.critic_repeat = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                                 hidden_dim, self.ensemble, self.noisy_net).to(device)
            self.critic_repeat.load_state_dict(torch.load(filename + "_critic"))
            self.encoder_repeat.load_state_dict(torch.load(filename + "_encoder"))
        if repeat_type > 0:
            self.hash_count = utils.HashingBonusEvaluator(dim_key=128,
                                                          obs_processed_flat_dim=feature_dim,
                                                          repeat_coefficient=repeat_coefficient)
        self.temp_cluster = temp_cluster
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.kmeans = KMeans(n_clusters=2,  init=np.array([[0.0], [1.0]]))


        # epsilon greedy
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_schedule = epsilon_schedule
        self.epsilon_zeta = epsilon_zeta
        self.repeat_steps = 0

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
        if self.train_dynamics_model != 0:
            self.dynamics_model.train(training)
            self.reward_model.train(training)

    def act(self, obs, step, eval_mode, last_action=None, pos=None):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        if self.pos_emb:
            position = torch.as_tensor([pos], device=self.device).unsqueeze(0)
            dist = self.actor(obs, stddev, pos=self.pe(position))
        else:
            dist = self.actor(obs, stddev)
        if eval_mode or self.test_model:
            action = dist.mean
        elif self.epsilon_greedy:
            if self.epsilon_zeta:
                if last_action is None:
                    self.repeat_steps = 0
                action = dist.mean
                current_epsilon = utils.schedule(self.epsilon_schedule, step)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
                elif self.repeat_steps > 0:
                    action = last_action
                    self.repeat_steps = self.repeat_steps-1
                    return action
                elif np.random.uniform() < current_epsilon:
                    action.uniform_(-1.0, 1.0)
                    self.repeat_steps = np.random.zipf(a=2)-1
            else:
                action = dist.mean
                current_epsilon = utils.schedule(self.epsilon_schedule, step)
                if step < self.num_expl_steps or np.random.uniform() < current_epsilon:
                    action.uniform_(-1.0, 1.0)
        elif self.noisy_net:
            action = dist.mean
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step, pos_feature):
        metrics = dict()

        target_all = []
        with torch.no_grad():
            for k in range(self.aug_K):
                stddev = utils.schedule(self.stddev_schedule, step)
                if self.pos_emb:
                    dist = self.actor(next_obs[k], stddev, pos=pos_feature)
                else:
                    dist = self.actor(next_obs[k], stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                if self.pos_emb:
                    target_Q_list = self.critic_target(next_obs[k], next_action, pos=pos_feature)
                else:
                    target_Q_list = self.critic_target(next_obs[k], next_action)
                two_Qs_index = np.random.choice(np.arange(self.critic.num_Qs), size=2, replace=False)
                target_V = torch.min(target_Q_list[two_Qs_index[0]], target_Q_list[two_Qs_index[1]])
                target_Q = reward + (discount * target_V)
                target_all.append(target_Q)
            avg_target_Q = sum(target_all)/self.aug_K

        critic_loss_all = []
        for k in range(self.aug_K):
            if self.pos_emb:
                Q_list = self.critic(obs[k], action, pos=pos_feature)
            else:
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
        if self.pos_emb:
            self.pos_emb_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        avg_critic_loss.backward()
        self.critic_opt.step()
        if self.pos_emb:
            self.pos_emb_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, pos_feature):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        if self.pos_emb:
            dist = self.actor(obs[0], stddev, pos=pos_feature)
        else:
            dist = self.actor(obs[0], stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Q1, Q2 = self.critic(obs[0], action)
        # Q = torch.min(Q1, Q2)

        if self.pos_emb:
            Q_list = self.critic(obs[0], action, pos=pos_feature)
        else:
            Q_list = self.critic(obs[0], action)
        Q = torch.min(torch.stack(Q_list), dim=0)[0]

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_dynamics_reward_model(self, obs, action, reward, next_obs):
        metrics = dict()
        if self.train_dynamics_model == 1:
            predicted_next_obs_feature = self.dynamics_model(self.encoder(obs), action)
            with torch.no_grad():
                next_obs_feature = self.dynamics_model.trunk(self.encoder(next_obs))
            dynamics_loss = torch.nn.functional.mse_loss(predicted_next_obs_feature, next_obs_feature)

            predicted_reward = self.reward_model(predicted_next_obs_feature)
            reward_loss = torch.nn.functional.mse_loss(predicted_reward, reward)

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

    def temporal_clustering(self, obs, next_obs):
        metrics = dict()
        obs_feature = self.critic.trunk(self.encoder(obs))
        next_obs_feature = self.critic.trunk(self.encoder(next_obs))
        cos_sim = self.cosine_sim(obs_feature, next_obs_feature)
        exp_cossim = torch.exp(cos_sim)

        with torch.no_grad():
            fitted = self.kmeans.fit(cos_sim.cpu().numpy().reshape(-1, 1))
            label = torch.tensor((fitted.labels_ == 0).reshape(-1, 1), device=self.device)
        loss = -torch.log(torch.sum(exp_cossim)/torch.sum(exp_cossim*label))

        # optimize dynamics model, reward model and encoder
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics['temp_loss'] = loss.item()

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

        if self.repeat_type > 0:
            batch = next(replay_iter)
            obs, action, reward, discount, repeat, next_obs, one_step_next_obs, one_step_reward = \
                utils.to_torch(batch, self.device)
            batch_size = obs.size(0)
            mask = (repeat == 1)
            obs = obs[mask]
            action = action[mask]
            reward = reward[mask]
            discount = discount[mask]
            next_obs = next_obs[mask]
            num_sample = obs.size(0)
            while num_sample < batch_size:
                new_batch = next(replay_iter)
                new_obs, new_action, new_reward, new_discount, new_repeat, new_next_obs,\
                new_one_step_next_obs, new_one_step_reward = \
                    utils.to_torch(new_batch, self.device)
                mask = (new_repeat == 1)
                obs = torch.vstack((obs, new_obs[mask]))
                action = torch.vstack((action, new_action[mask]))
                reward = torch.vstack((reward, new_reward[mask]))
                discount = torch.vstack((discount, new_discount[mask]))
                next_obs = torch.vstack((next_obs, new_next_obs[mask]))
                num_sample = obs.size(0)
            obs = obs[:batch_size]
            action = action[:batch_size]
            reward = reward[:batch_size]
            discount = discount[:batch_size]
            next_obs = next_obs[:batch_size]
            # update hash count
            with torch.no_grad():
                if self.load_folder != 'None':
                    feature = (self.critic_repeat.trunk(self.encoder_repeat(obs.float()))).cpu().numpy()
                else:
                    feature = (self.critic.trunk(self.encoder(obs.float()))).cpu().numpy()
                self.hash_count.fit_before_process_samples(feature)
        else:
            batch = next(replay_iter)
            obs, action, reward, discount, repeat, traj_index, next_obs, one_step_next_obs, one_step_reward = \
                utils.to_torch(batch, self.device)

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

        if self.pos_emb:
            pos_feature = self.pe(traj_index)
        # update critic
        metrics.update(
            self.update_critic(obs_all, action, reward, discount, next_obs_all, step, pos_feature))

        # update dynamics and reward model
        if self.train_dynamics_model != 0:
            metrics.update(self.update_dynamics_reward_model(obs.float(), action, one_step_reward,
                                                             one_step_next_obs.float()))
        # temporal clustering
        if self.temp_cluster:
            metrics.update(self.temporal_clustering(obs.float(), one_step_next_obs.float()))

        # update actor
        for k in range(self.aug_K):
            obs_all[k] = obs_all[k].detach()
        pos_feature = pos_feature.detach()
        metrics.update(self.update_actor(obs_all, step, pos_feature))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        if self.noisy_net:
            self.critic.reset_noise()
            self.critic_target.reset_noise()

            self.actor.reset_noise()

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