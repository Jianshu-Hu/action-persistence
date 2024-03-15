# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
import math

import dmc
import utils
from logger import Logger
# from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, work_dir_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.work_dir = str(work_dir_spec)
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.work_dir,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_buffer = hydra.utils.instantiate(self.cfg.replay_buffer, data_specs=data_specs)

        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        total_smoothness = 0
        while eval_until_episode(episode):
            last_action = None
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                if last_action is None:
                    last_action = action
                else:
                    smoothness = np.mean(np.square(action-last_action))
                    total_smoothness += smoothness
                    last_action = action
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('episode_smoothness', total_smoothness / episode)

        return total_reward / episode

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        save_every_step = utils.Every(self.cfg.save_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        repeat_prob_record_list = []
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                evaluated_reward = self.eval()

            # try to save the model
            if self.cfg.save_model:
                if save_every_step(self.global_step):
                    save_dir = str(self.work_dir)+'/saved_model'
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    self.agent.save(save_dir+'/ar_'+str(self.cfg.action_repeat)+'_step_'+str(self.global_step))

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.repeat_type == 1:
                    # hash count (state)
                    obs_torch = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0)
                    feature = (self.agent.critic.trunk(self.agent.encoder(obs_torch))).cpu().numpy()
                    if self.global_step == 0:
                        action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    else:
                        repeat_prob = self.agent.hash_count.predict(feature)
                        if np.random.uniform() < repeat_prob:
                            action = last_action
                        else:
                            action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                        repeat_prob_record_list.append(repeat_prob)
                        if self.global_step % 5000 == 0:
                            np.savez(str(self.work_dir) + '/repeat_prob.npz',
                                     repeat_prob=np.array(repeat_prob_record_list))
                elif self.cfg.repeat_type == 2:
                    # hash count (state-action)
                    action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    obs_torch = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0)
                    feature = (self.agent.critic.trunk(self.agent.encoder(obs_torch))).cpu().numpy()
                    state_action_feature = np.concatenate((feature, np.expand_dims(action, axis=0)), axis=1)
                    repeat_prob = self.agent.hash_count.predict(state_action_feature)
                    if self.global_step == 0:
                        action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    elif np.random.uniform() < repeat_prob:
                        action = last_action
                elif self.cfg.repeat_type == 3:
                    # RND
                    if len(self.agent.initial_loss_list) < self.agent.initial_loss_max:
                        action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    else:
                        with torch.no_grad():
                            obs_torch = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0)
                            predict_feature, target_feature = self.agent.rnd(obs_torch.float())
                            loss = torch.nn.functional.mse_loss(predict_feature, target_feature)
                            repeat_prob = loss.item()/\
                                          (sum(self.agent.initial_loss_list)/len(self.agent.initial_loss_list))
                            repeat_prob_record_list.append(repeat_prob)
                        if np.random.uniform() < repeat_prob:
                            action = last_action
                        else:
                            action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                        if self.global_step % 5000 == 0:
                            np.savez(str(self.work_dir) + '/repeat_prob.npz',
                                     repeat_prob=np.array(repeat_prob_record_list))
                elif self.cfg.repeat_type == 4:
                    # use the std/mean of ensemble critic for the repeat probability
                    action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    coefficient = 25.0
                    with torch.no_grad():
                        obs_torch = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0)
                        action_torch = torch.as_tensor(action, device=self.device).unsqueeze(0)
                        Q_ensemble = torch.stack(self.agent.critic(
                            self.agent.encoder(obs_torch.float()), action_torch), dim=0).view(-1)
                        Q_mean = torch.mean(Q_ensemble).item()
                        if Q_mean < 0:
                            repeat_prob = 1.0
                        else:
                            repeat_prob = coefficient*torch.std(Q_ensemble).item()/Q_mean
                        repeat_prob_record_list.append(repeat_prob)
                    if self.global_step == 0:
                        action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    elif np.random.uniform() < repeat_prob:
                        action = last_action
                    if self.global_step % 5000 == 0:
                        np.savez(str(self.work_dir) + '/repeat_prob.npz',
                                 repeat_prob=np.array(repeat_prob_record_list))
                elif self.cfg.repeat_type == 5:
                    # use the std of sampled Q for the repeat probability
                    action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    number_samples = 10
                    coefficient = 0.1
                    noise_coefficient = 0.2
                    # norm_dist_action = torch.distributions.normal.Normal(torch.as_tensor(action, device=self.device),
                    #                             variance_coefficient*torch.ones(action.shape[0], device=self.device))
                    # action_list = []
                    # for i in range(number_samples):
                    #     sampled_action = norm_dist_action.sample()
                    #     action_list.append(sampled_action)
                    # action_torch = torch.clip(torch.stack(action_list), -1, 1)

                    action_torch = torch.as_tensor(action, device=self.device).unsqueeze(0)
                    noise = noise_coefficient*(torch.rand((number_samples, action.shape[0]))*2-1.0).to(self.device)
                    noisy_action = action_torch.repeat(number_samples, 1)+noise
                    action_torch = torch.concat((noisy_action, action_torch), 0)
                    action_torch = torch.clip(action_torch, -1, 1)

                    # action_torch = torch.zeros([number_samples, self.cfg.agent.action_shape[0]],
                    #                            dtype=torch.float32, device=self.device)
                    # action_torch.uniform_(-1.0, 1.0)
                    with torch.no_grad():
                        obs_torch = (torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0))
                        obs_feature = (self.agent.encoder(obs_torch.float())).repeat(number_samples+1, 1)
                        Q_list = self.agent.critic(obs_feature, action_torch)
                        Q_samples = torch.min(torch.stack(Q_list), dim=0)[0]
                        if torch.mean(Q_samples) < 0:
                            repeat_prob = 1.0
                        else:
                            repeat_prob = coefficient/(torch.mean(Q_samples)*torch.std(Q_samples))
                            repeat_prob_record_list.append(repeat_prob.item())
                    if self.global_step == 0:
                        action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                    elif np.random.uniform() < repeat_prob:
                        action = last_action
                    if self.global_step % 5000 == 0:
                        np.savez(str(self.work_dir) + '/repeat_prob.npz',
                                 repeat_prob=np.array(repeat_prob_record_list))
                else:
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.num_updates):
                    metrics = self.agent.update(self.replay_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            last_action = copy.deepcopy(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()