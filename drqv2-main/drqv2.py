# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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

        self.loaded_actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim)

        self.apply(utils.weight_init)

    def load(self, filename):
        # load the actor
        self.loaded_actor.load_state_dict(torch.load(filename + "_actor"))
        self.trunk.load_state_dict(self.loaded_actor.trunk.state_dict())
        self.loaded_policy.load_state_dict(self.loaded_actor.policy.state_dict())

        for param in self.loaded_policy.parameters():
            param.requires_grad = False

        # copy policy
        self.copied_policy.load_state_dict(self.trained_policy.state_dict())

        for param in self.copied_policy.parameters():
            param.requires_grad = False

    def forward_mu_std(self, obs, std):
        h = self.trunk(obs)

        loaded_mu = torch.tanh(self.loaded_policy(h))
        loaded_std = torch.ones_like(loaded_mu) * std

        trained_mu = torch.tanh(self.trained_policy(h))
        trained_std = torch.ones_like(trained_mu) * std

        copied_mu = torch.tanh(self.copied_policy(h))
        copied_std = torch.ones_like(copied_mu) * std
        return loaded_mu+trained_mu-copied_mu, loaded_std+trained_std-copied_std

    def forward(self, obs, std):
        final_mu, final_std = self.forward_mu_std(obs, std)

        dist = utils.TruncatedNormal(final_mu, final_std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


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

        self.loaded_critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim)

        self.apply(utils.weight_init)

    def load(self, filename):
        # load the critic
        self.loaded_critic.load_state_dict(torch.load(filename + "_critic"))
        self.trunk.load_state_dict(self.loaded_critic.trunk.state_dict())
        self.loaded_Q1.load_state_dict(self.loaded_critic.Q1.state_dict())
        self.loaded_Q2.load_state_dict(self.loaded_critic.Q2.state_dict())

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


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, work_dir, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 aug_K, aug_type, add_KL_loss, tangent_prop, train_dynamics_model,
                 time_reflection, load_model, task_name, test_model):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

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
        self.time_reflection = time_reflection
        if self.train_dynamics_model:
            self.dynamics_model = DynamicsModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.critic.trunk).to(device)
            self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)

            self.reward_model = RewardModel(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)


        # load model
        self.work_dir = work_dir
        self.load_model = load_model
        self.test_model = test_model
        self.task_name = task_name
        if load_model != 'none':
            self.extend_model = False
            if self.extend_model:
                self.critic = ExCritic(self.encoder.repr_dim, action_shape, feature_dim,
                                       hidden_dim).to(device)
                self.critic_target = ExCritic(self.encoder.repr_dim, action_shape,
                                              feature_dim, hidden_dim).to(device)
                # optimizers
                self.critic_opt = torch.optim.Adam(list(self.critic.trunk.parameters()) +
                                                   list(self.critic.trained_Q1.parameters()) + list(
                    self.critic.trained_Q2.parameters()), lr=lr)

                self.actor = ExActor(self.encoder.repr_dim, action_shape, feature_dim,
                                       hidden_dim).to(device)
                # optimizers
                self.actor_opt = torch.optim.Adam(list(self.actor.trunk.parameters()) +
                                                   list(self.actor.trained_policy.parameters()), lr=lr)

            self.load(self.work_dir+'/../../../saved_model/' + task_name + '/' + load_model)
            print('load model from: ')
            print(self.work_dir+'/../../../saved_model/' + task_name + '/' + load_model)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        if self.train_dynamics_model:
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

    def update_critic(self, obs, action, reward, discount, next_obs, step, obs_original, one_step_next_obs_original):
        metrics = dict()

        target_all = []
        with torch.no_grad():
            for k in range(self.aug_K):
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs[k], stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Q1, target_Q2 = self.critic_target(next_obs[k], next_action)
                target_V = torch.min(target_Q1, target_Q2)
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
                Q1, Q2 = self.critic(obs[k], action)
                critic_loss = F.mse_loss(Q1, avg_target_Q) + F.mse_loss(Q2, avg_target_Q)
                critic_loss_all.append(critic_loss)
            avg_critic_loss = sum(critic_loss_all) / self.aug_K

        # if self.time_reflection:
        #     with torch.no_grad():
        #         # reflect the obs, action, next_obs
        #         reflected_obs = self.encoder(self.time_reflect_obs(self.aug(one_step_next_obs_original)))
        #         reflected_next_obs = self.encoder(self.time_reflect_obs(self.aug(obs_original)))
        #         reflected_action = -action
        #
        #         # target Q for reflected next obs
        #         dist = self.actor(reflected_next_obs, stddev)
        #         next_action = dist.sample(clip=self.stddev_clip)
        #         target_Q1, target_Q2 = self.critic_target(reflected_next_obs, next_action)
        #         target_V = torch.min(target_Q1, target_Q2)
        #         target_Q = reward + (discount * target_V)
        #
        #     Q1, Q2 = self.critic(reflected_obs, reflected_action)
        #     reflect_time_critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        #     avg_critic_loss += reflect_time_critic_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = avg_critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        avg_critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, obs_original):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs[0], stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs[0], action)
        Q = torch.min(Q1, Q2)

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

        return metrics

    def time_reflect_obs(self, obs):
        stacked_num = int(obs.size(1)/3)
        reflected_obs = torch.zeros_like(obs).to(obs.device)
        for i in range(stacked_num):
            reflected_obs[:, 3 * i:3 * (i + 1), :, :] = obs[:, obs.size(1)-3 * (i + 1):obs.size(1)-3 * i, :, :]
        return reflected_obs

    def update_dynamics_reward_model(self, obs, action, reward, next_obs):
        metrics = dict()

        predicted_next_obs_feature = self.dynamics_model(self.encoder(obs), action)
        with torch.no_grad():
            next_obs_feature = self.dynamics_model.trunk(self.encoder(next_obs))
        dynamics_loss = torch.nn.functional.mse_loss(predicted_next_obs_feature, next_obs_feature)

        # if self.time_reflection:
        #     reflected_obs = self.time_reflect_obs(next_obs)
        #     reflected_next_obs = self.time_reflect_obs(obs)
        #     reflected_action = -action
        #     reflected_predicted_next_obs_feature = self.dynamics_model(self.encoder(reflected_obs), reflected_action)
        #     with torch.no_grad():
        #         reflected_next_obs_feature = self.dynamics_model.trunk(self.encoder(reflected_next_obs))
        #     dynamics_loss += torch.nn.functional.mse_loss(reflected_predicted_next_obs_feature,
        #                                                   reflected_next_obs_feature)
        #     dynamics_loss = dynamics_loss/2

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
            self.update_critic(obs_all, action, reward, discount, next_obs_all, step,
                               obs.float(), one_step_next_obs.float()))

        # update dynamics and reward model
        if self.train_dynamics_model:
            with torch.no_grad():
                one_step_next_obs_aug = self.aug(one_step_next_obs.float())

            metrics.update(self.update_dynamics_reward_model(obs_aug, action, one_step_reward, one_step_next_obs_aug))

        # update actor
        for k in range(self.aug_K):
            obs_all[k] = obs_all[k].detach()
        metrics.update(self.update_actor(obs_all, step, obs.float()))

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

        if self.train_dynamics_model:
            torch.save(self.dynamics_model.state_dict(), filename + "_dynamics_model")
            torch.save(self.dynamics_opt.state_dict(), filename + "_dynamics_optimizer")

            torch.save(self.reward_model.state_dict(), filename + "_reward_model")
            torch.save(self.reward_opt.state_dict(), filename + "_reward_optimizer")

    def load(self, filename):
        if self.extend_model:
            self.critic.load(filename)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor.load(filename)
        else:
            self.critic.load_state_dict(torch.load(filename + "_critic"))
            self.critic_opt.load_state_dict(torch.load(filename + "_critic_optimizer"))
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor.load_state_dict(torch.load(filename + "_actor"))
            self.actor_opt.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.encoder_opt.load_state_dict(torch.load(filename + "_encoder_optimizer"))

        if self.train_dynamics_model:
            self.dynamics_model.load_state_dict(torch.load(filename + "_dynamics_model"))
            self.dynamics_opt.load_state_dict(torch.load(filename + "_dynamics_optimizer"))

            self.reward_model.load_state_dict(torch.load(filename + "_reward_model"))
            self.reward_opt.load_state_dict(torch.load(filename + "_reward_optimizer"))