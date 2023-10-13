#  https://github.com/tuzzer/ai-gym/blob/master/maze_2d/maze_2d_q_learning.py
import sys
import numpy as np
import math
import random
import time
import os
import argparse

import gym
import gym_maze


def simulate():

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0

    # Render tha maze
    if RENDER_MAZE:
        env.render()

    # Record total time steps for solving the tasks
    total_t = 0
    dir = os.path.dirname(os.path.abspath(__file__))
    folder = dir+'/runs/'+time.strftime('%Y-%m-%d-%H:%M:%S-', time.localtime())+ENV+'-seed-'+str(args.seed)
    os.mkdir(folder)
    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate)

            # execute the action
            obv, reward, done, info = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # update with time reflection
            if TIME_REFLECTION:
                if not done and (state_0 != state):
                    best_q = np.amax(q_table[state_0])
                    q_table[state + (inverse_action(action),)] += learning_rate *\
                                  (reward + discount_factor * (best_q) - q_table[state + (inverse_action(action),)])
            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))
        total_t += t
        header = "episode, total_timesteps, reward, num_success, time_reflection"
        np.savetxt(folder+'/eval.csv',
                   np.array([episode, total_t, total_reward, num_streaks, TIME_REFLECTION]).reshape([1, 5]),
                   delimiter=",", header=header)
        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            print("total time steps = %d" % total_t)
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action

def inverse_action(action):
    inv_action_list = [1, 0, 3, 2]
    inv_action = inv_action_list[action]
    return inv_action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def set_seed_everywhere(seed):
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--env', type=str, default='maze-sample-5x5-v0', help='type of environment')
    parser.add_argument('--time_inv', type=int, default=0, help='use time-reflection or not')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    set_seed_everywhere(args.seed)

    ENV = args.env
    RENDER_MAZE = False
    ENABLE_RECORDING = False
    # Initialize the "maze" environment
    if RENDER_MAZE:
        env = gym.make(ENV)
    else:
        env = gym.make(ENV, enable_render=False)
    env.seed(args.seed)

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 50000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 50
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0

    '''
    Creating a Q-Table for each state-action pair
    '''
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    '''
    Learning related arguments
    '''
    TIME_REFLECTION = bool(args.time_inv)
    print('use time reflection: ' + str(TIME_REFLECTION))

    '''
    Begin simulation
    '''
    recording_folder = "/tmp/maze_q_learning"

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    simulate()

    if ENABLE_RECORDING:
        env.monitor.close()