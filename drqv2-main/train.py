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
from replay_buffer import ReplayBufferStorage, make_replay_loader
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

        if self.cfg.load_model != 'none':
            self.replay_storage = ReplayBufferStorage(data_specs,
                                                      self.work_dir / 'buffer')

            self.replay_loader = make_replay_loader(
                self.work_dir / 'buffer', self.cfg.replay_buffer_size,
                int(self.cfg.batch_size/2), self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.cfg.test_model,
                self.cfg.time_ssl_K, self.cfg.dyn_prior_K)
            self._replay_iter = None

            # create a replay buffer for saving transitions for larger action repeat
            self.old_replay_storage = ReplayBufferStorage(data_specs,
                                                      self.work_dir / 'old_buffer')

            self.old_replay_loader = make_replay_loader(
                self.work_dir / 'old_buffer', self.cfg.load_num_frames,
                int(self.cfg.batch_size/2), self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot, 1, self.cfg.discount, self.cfg.test_model,
                self.cfg.time_ssl_K, self.cfg.dyn_prior_K)
            self._old_replay_iter = None
        elif self.cfg.transfer:
            self.replay_storage = ReplayBufferStorage(data_specs,
                                                      self.work_dir / 'buffer')
            self.replay_loader = make_replay_loader(
                self.work_dir / 'buffer', self.cfg.replay_buffer_size,
                self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot, int(self.cfg.nstep*2), self.cfg.discount, self.cfg.test_model,
                self.cfg.time_ssl_K, self.cfg.dyn_prior_K)
            self._replay_iter = None
        else:
            self.replay_storage = ReplayBufferStorage(data_specs,
                                                      self.work_dir / 'buffer')

            self.replay_loader = make_replay_loader(
                self.work_dir / 'buffer', self.cfg.replay_buffer_size,
                self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.cfg.test_model,
                self.cfg.time_ssl_K, self.cfg.dyn_prior_K)
            self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def old_replay_iter(self):
        if self._old_replay_iter is None:
            self._old_replay_iter = iter(self.old_replay_loader)
        return self._old_replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
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

        return total_reward / episode

    def eval_large_repeat_policy(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            episode_step = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if episode_step % 2 == 0:
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    else:
                        action = last_action
                time_step = self.eval_env.step(action)
                last_action = copy.deepcopy(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                episode_step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        return total_reward / episode

    def eval_loaded_policy(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            episode_step = 0
            time_step = self.eval_env.reset()
            three_time_steps = [copy.deepcopy(time_step), copy.deepcopy(time_step), copy.deepcopy(time_step)]
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if episode_step % 2 == 0:
                        action = self.agent.loaded_policy_act(three_time_steps[0].observation,
                                                              three_time_steps[1].observation,
                                                              three_time_steps[2].observation,
                                                              self.global_step, eval_mode=True)
                    else:
                        action = last_action
                time_step = self.eval_env.step(action)
                last_action = copy.deepcopy(action)
                three_time_steps.pop(0)
                three_time_steps.append(time_step)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                episode_step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        return total_reward / episode

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        load_until_step = utils.Until(self.cfg.load_num_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        save_every_step = utils.Every(self.cfg.save_every_frames,
                                      self.cfg.action_repeat)
        transfer_until_step = utils.Until(self.cfg.transfer_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        # record previous steps for getting the action from loaded policy
        three_time_steps = [copy.deepcopy(time_step), copy.deepcopy(time_step), copy.deepcopy(time_step)]
        if self.cfg.load_model != 'none':
            self.old_replay_storage.add(time_step)
        else:
            self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
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
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                three_time_steps = [copy.deepcopy(time_step), copy.deepcopy(time_step), copy.deepcopy(time_step)]
                if self.cfg.load_model != 'none' and load_until_step(self.global_step):
                    self.old_replay_storage.add(time_step)
                else:
                    self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to pretrain the actor-critic
            if self.cfg.load_model != 'none' and\
                    (self.global_step == (self.cfg.load_num_frames // self.cfg.action_repeat)):
                print('start pretrain')
                self.agent.pretrain(self.old_replay_iter)

            # test pretraining
            # if self.cfg.load_model != 'none' and\
            #         (self.global_step == (self.cfg.load_num_frames // self.cfg.action_repeat)):
            #     print('start pretrain')
            #     for i in range(100):
            #         self.agent.pretrain(self.old_replay_iter)
            #         evaluated_reward = self.eval()
            #         print('pretrain iter: ' + str(i))
            #         print('eval_reward: ' + str(evaluated_reward))
            #     raise ValueError('test')

            # change the nstep
            # if self.cfg.transfer and \
            #         (self.global_step == (self.cfg.transfer_frames // self.cfg.action_repeat)):
                # self.replay_loader.dataset.change_nstep(self.cfg.nstep)
                # reset the agent
                # self.agent.reset()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.load_model != 'none' and load_until_step(self.global_step):
                    loaded_policy_reward = self.eval_loaded_policy()
                elif self.cfg.transfer and transfer_until_step(self.global_step):
                    large_repeat_reward = self.eval_large_repeat_policy()
                else:
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
                if self.cfg.load_model != 'none' and load_until_step(self.global_step):
                    if episode_step % 2 == 0:
                        action = self.agent.loaded_policy_act(three_time_steps[0].observation,
                                                              three_time_steps[1].observation,
                                                              three_time_steps[2].observation,
                                                              self.global_step, eval_mode=False)
                    else:
                        action = last_action
                elif self.cfg.transfer:
                    if transfer_until_step(self.global_step):
                        if episode_step % 2 == 0:
                            action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                        else:
                            action = last_action
                    else:
                        action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
                else:
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)

            # try to update the agent
            if self.cfg.load_model != 'none':
                if load_until_step(self.global_step):
                    pass
                elif seed_until_step(self.global_step-(self.cfg.load_num_frames // self.cfg.action_repeat)):
                    pass
                else:
                    for _ in range(self.cfg.num_updates):
                        metrics = self.agent.update(self.replay_iter, self.global_step, self.old_replay_iter)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
            elif not seed_until_step(self.global_step):
                for _ in range(self.cfg.num_updates):
                    metrics = self.agent.update(self.replay_iter, self.global_step, None)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            last_action = copy.deepcopy(action)
            three_time_steps.pop(0)
            three_time_steps.append(time_step)
            episode_reward += time_step.reward
            if self.cfg.load_model != 'none' and load_until_step(self.global_step):
                self.old_replay_storage.add(time_step)
            else:
                self.replay_storage.add(time_step)
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