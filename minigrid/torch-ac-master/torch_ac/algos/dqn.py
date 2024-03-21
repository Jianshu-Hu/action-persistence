import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import random
import numpy as np


class ReplayMemory(object):
    '''
        Memory buffer for Experience Replay
    '''
    def __init__(self, capacity, obs_shape):
        '''
            Initialize a buffer containing max_size experiences
        '''
        self.obs_shape = obs_shape
        self.capacity = capacity
        self.obs = np.zeros([self.capacity, *self.obs_shape[:]], dtype=np.float32)
        self.next_obs = np.zeros([self.capacity, *self.obs_shape[:]], dtype=np.float32)
        self.act = np.zeros([self.capacity, 1], dtype=np.float32)
        self.rew = np.zeros([self.capacity], dtype=np.float32)
        self.position = 0
        self.full = False

    def add(self, experience):
        '''
            Add an experience to the buffer
        '''
        self.obs[self.position] = experience[0].cpu().numpy()
        self.act[self.position] = experience[1]
        self.next_obs[self.position] = experience[2].cpu().numpy()
        self.rew[self.position] = experience[3][0]
        if self.position+1 == self.capacity:
            self.full = True
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
            Sample a batch of experiences from the buffer
        '''
        if self.full:
            indices = np.random.choice(self.capacity, size=batch_size)
        else:
            indices = np.random.choice(self.position, size=batch_size)
        return self.obs[indices], self.act[indices], self.next_obs[indices], self.rew[indices]


class DQNAlgo():
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, eval_envs, Q_network, target_network, device=None, frames_per_proc=1, discount=0.99, lr=0.001,
                 adam_eps=1e-8, epochs=4, batch_size=256, preprocess_obss=None,
                 total_frames=100000, init_epsilon=1.0, final_epsilon=0.1, zeta_epsilon=False, simhash_repeat=False,
                 target_freq=100, buffer_size=100000, init_expl=1000):

        self.env = ParallelEnv(envs)
        self.eval_env = ParallelEnv(eval_envs)
        self.Q_network = Q_network
        self.target_network = target_network
        self.device = device
        self.discount = discount
        self.lr = lr
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.n_actions = envs[0].action_space.n

        # Configure model
        self.Q_network.to(self.device)
        self.Q_network.train()
        self.target_network.to(self.device)
        self.target_network.train()

        self.epochs = epochs
        self.batch_size = batch_size

        # exploration
        self.total_frames = total_frames
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.zeta_epsilon = zeta_epsilon
        self.simhash_repeat = simhash_repeat
        if self.simhash_repeat:
            self.obs = self.env.reset()
            tmp_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                feature = self.Q_network.forward_emb(tmp_obs)
            self.simhash_count = HashingBonusEvaluator(dim_key=128, obs_processed_flat_dim=feature.size(1))
        self.init_expl = init_expl

        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr, eps=adam_eps)
        self.update_target = target_freq
        if frames_per_proc is None:
            self.frames_per_proc = 1
        else:
            self.frames_per_proc = frames_per_proc

        self.obs = self.env.reset()
        self.done = (False,)
        self.last_action = None
        self.repeat_num = 0

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        self.replay_buffer = ReplayMemory(capacity=buffer_size, obs_shape=preprocessed_obs.image.shape[1:])

    def eval(self, num_eval_episode=20):
        sum_reward = []
        for i in range(num_eval_episode):
            obs = self.eval_env.reset()
            done = (False,)
            while not done[0]:
                preprocessed_obs = self.preprocess_obss(obs, device=self.device)
                with torch.no_grad():
                    Q = self.Q_network(preprocessed_obs)[0]
                    action = torch.argmax(Q).item()

                next_obs, reward, terminated, truncated, _ = self.eval_env.step([action])
                sum_reward.append(reward[0])
                done = tuple(a | b for a, b in zip(terminated, truncated))
                obs = next_obs
        return sum(sum_reward)/num_eval_episode

    def collect_experiences(self, num_frames):
        # interact with env
        if self.done[0]:
            self.obs = self.env.reset()
            self.done = (False,)
            self.last_action = None
            self.repeat_num = 0
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        # choose action
        current_epsilon = self.init_epsilon-(num_frames/self.total_frames)*(self.init_epsilon-self.final_epsilon)
        if num_frames <= self.init_expl:
            action = random.randrange(self.n_actions)
        else:
            if self.zeta_epsilon:
                if self.repeat_num == 0:
                    if random.random() > current_epsilon:
                        with torch.no_grad():
                            Q = self.Q_network(preprocessed_obs)[0]
                            action = torch.argmax(Q).item()
                    else:
                        action = random.randrange(self.n_actions)
                        self.repeat_num = np.random.zipf(a=2)-1
                else:
                    action = self.last_action
                    self.repeat_num = self.repeat_num-1
            elif self.simhash_repeat:
                with torch.no_grad():
                    feature = self.Q_network.forward_emb(preprocessed_obs)
                repeat_prob = self.simhash_count.predict(feature.cpu().numpy())
                if random.random() > repeat_prob or self.last_action is None:
                    if random.random() > current_epsilon:
                        with torch.no_grad():
                            Q = self.Q_network(preprocessed_obs)[0]
                            action = torch.argmax(Q).item()
                    else:
                        action = random.randrange(self.n_actions)
                else:
                    action = self.last_action
            else:
                if random.random() > current_epsilon:
                    with torch.no_grad():
                        Q = self.Q_network(preprocessed_obs)[0]
                        action = torch.argmax(Q).item()
                else:
                    action = random.randrange(self.n_actions)
        self.last_action = action
        next_obs, reward, terminated, truncated, _ = self.env.step([action])
        self.done = tuple(a | b for a, b in zip(terminated, truncated))

        preprocessed_next_obs = self.preprocess_obss(next_obs, device=self.device)
        transition = [preprocessed_obs.image, action, preprocessed_next_obs.image, reward]
        self.replay_buffer.add(transition)
        self.obs = next_obs

        # update
        if num_frames > self.init_expl and num_frames % self.frames_per_proc == 0:
            self.update_parameters(num_frames)

        num_frames += 1
        return num_frames

    def update_parameters(self, num_frames):
        # update network
        for _ in range(self.epochs):
            # update critic
            obs, actions, next_obs, rewards = self.replay_buffer.sample(self.batch_size)
            obs = torch.from_numpy(obs).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            next_obs = torch.from_numpy(next_obs).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            if self.simhash_repeat:
                # update hash table
                with torch.no_grad():
                    feature = self.Q_network.forward_emb(obs, train=True)
                    self.simhash_count.fit_before_process_samples(feature.cpu().numpy())
            Q = self.Q_network(obs, train=True)[torch.arange(actions.size(0)), torch.squeeze(actions, 1).to(torch.long)]
            with torch.no_grad():
                target_Q = rewards+self.discount*self.target_network(next_obs, train=True).max(dim=1)[0]

            # compute loss
            loss = F.smooth_l1_loss(Q, target_Q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update target network if necessary
            if num_frames % self.update_target == 0:
                self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())


class HashingBonusEvaluator(object):
    """Hash-based count bonus for exploration.

    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, X., Duan, Y., Schulman, J., De Turck, F., and Abbeel, P. (2017).
    #Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in Neural Information Processing Systems (NIPS)
    """

    def __init__(self, dim_key=128, obs_processed_flat_dim=None, bucket_sizes=None):
        # Hashing function: SimHash
        if bucket_sizes is None:
            # Large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        self.projection_matrix = np.random.normal(size=(obs_processed_flat_dim, dim_key))

    def compute_keys(self, obss):
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, obss):
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_hash(self, obss):
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.query_hash(obss)
        self.inc_hash(obss)

    def predict(self, obs):
        counts = self.query_hash(obs)
        return 1. / np.maximum(1., np.sqrt(counts))
