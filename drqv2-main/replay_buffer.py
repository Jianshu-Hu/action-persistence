# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import abc
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, new_obs=None):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            if new_obs is not None and spec.name == 'observation':
                self._current_episode[spec.name].append(new_obs)
            else:
                self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class IterableReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, test_model, time_ssl_K, dyn_prior_K):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

        # load the whole episode for testing the model
        self._test_model = test_model
        # sample K steps obs for calculating temporal ssl
        self._time_ssl_K = time_ssl_K
        self._dyn_prior_K = dyn_prior_K

    def change_nstep(self, nstep):
        self._nstep = nstep

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            if early_eps_fn.exists():
                early_eps_fn.unlink()
            # early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            if eps_fn.exists():
                eps_fn.unlink()
            # eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        if self._test_model:
            return episode['observation'], episode['action'], episode['reward']
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        episode_return = np.zeros_like(episode['reward'][idx])
        return_discount = np.ones_like(episode['discount'][idx])
        for i in range(episode_len(episode)-idx):
            step_reward = episode['reward'][idx + i]
            episode_return += return_discount * step_reward
            return_discount *= episode['discount'][idx + i] * self._discount

        one_step_next_obs = episode['observation'][idx]
        one_step_reward = episode['reward'][idx]

        # next k step for temporal ssl
        num_next_steps_1 = int(2 * self._time_ssl_K - 1)
        num_next_steps_2 = self._dyn_prior_K
        num_next_steps = max(num_next_steps_1, num_next_steps_2, 1)
        idx_temporal_ssl = np.random.randint(0, episode_len(episode) - num_next_steps + 1) + 1
        next_K_step_obs = np.zeros([num_next_steps, obs.shape[0], obs.shape[1], obs.shape[2]], dtype=obs.dtype)
        for k in range(num_next_steps):
            next_K_step_obs[k, :, :, :] = episode['observation'][idx_temporal_ssl - 1 + k]
        return (obs, action, reward, discount, next_obs, np.array([idx]), one_step_next_obs, one_step_reward, next_K_step_obs,
                np.array([idx_temporal_ssl-1]), episode_return)

    def __iter__(self):
        while True:
            yield self._sample()


class AbstractReplayBuffer(abc.ABC):
   @abc.abstractmethod
   def add(self, time_step, repeat):
       pass

   @abc.abstractmethod
   def __next__(self, ):
       pass

   @abc.abstractmethod
   def __len__(self, ):
       pass


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = IterableReplayBuffer(replay_dir,
                                    max_size_per_worker,
                                    num_workers,
                                    nstep,
                                    discount,
                                    fetch_every=1000,
                                    save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
