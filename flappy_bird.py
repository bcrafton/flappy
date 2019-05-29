
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mini_batch_size', type=int, default=512)
parser.add_argument('--name', type=str, default="flappy")
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


import numpy as np
import tensorflow as tf
import cv2
import gym
import gym_ple
from collections import deque
import random
# import matplotlib.pyplot as plt

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.BatchNorm import BatchNorm

from lib.Activation import Activation
from lib.Activation import Relu
from lib.Activation import Linear

total_episodes = int(1e5)
epsilon_init = 0.1
decay_rate = epsilon_init / (1.0 * total_episodes)

####################################

def returns_advantages (replay_buffer, next_value, gamma=0.99, lam=0.95):
    rewards = [rb['r'] for rb in replay_buffer]
    values = [rb['v'] for rb in replay_buffer] + [next_value]
    dones = [rb['d'] for rb in replay_buffer]

    gae = 0
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    for t in reversed(range(len(replay_buffer))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return returns, advantages

####################################

class FlappyBirdEnv:
    def __init__(self):
        self.env = gym.make('FlappyBird-v0')
        self.env.seed(np.random.randint(0, 100000))
        self.total_reward = 0.0
        self.total_step = 0
        self.state = None

    def reset(self):
        self.total_reward = 0.0
        self.total_step = 0
        
        frame = self.env.reset()
        frame = self._process(frame)
        self.state = deque([frame] * 4, maxlen=4)
        
        return np.stack(self.state, axis=2)

    def step(self, action):
        next_frame, reward, done, _ = self.env.step(action)
        reward = self._reward_shaping(reward)        
        next_frame = self._process(next_frame)
        self.state.append(next_frame)
        return np.stack(self.state, axis=2), reward, done

    def _reward_shaping(self, reward):
        if  reward > 0.0:
            return 1.0
        elif reward < 0.0:
            return -1.0
        else:
            return 0.01

    def _process(self, state):
        output = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        output = output[:410, :]
        output = cv2.resize(output, (80, 80))
        output = output / 255.0
        return output

####################################

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

####################################

model = PPOModel(nbatch=64, nclass=2, epsilon=0.1, decay=decay_rate)
replay_buffer = []
env = FlappyBirdEnv()
state = env.reset()

for e in range(total_episodes):
    
    #####################################

    replay_buffer = []
    for _ in range(args.mini_batch_size):

        '''
        q_value = predict1.eval(feed_dict={s1 : [state]})
        q_value = np.squeeze(q_value)
        
        if np.random.rand() < epsilon:
            action_idx = env.env.action_space.sample()
        else:
            action_idx = np.argmax(q_value)
            
        value = q_value[action_idx]
        '''
        
        value, action = model.predict(state)
        
        ################################

        next_state, reward, done = env.step(action_idx)

        if done and env.total_step >= 10000:
            next_value, next_action = model.predict(next_state)
            next_value = np.max(next_q_value)
            reward += 0.99 * next_value
        
        replay_buffer.append({'s':state, 'v': value, 'a':action, 'r':reward, 'd':done})
        state = next_state
        
        if done:
            state = env.reset()

    next_value, next_action = model.predict(next_state)
    next_value = np.max(next_q_value)
    rets, advs = returns_advantages(replay_buffer, next_value)
    
    #####################################

    states = [d['s'] for d in replay_buffer]
    # rewards = rets
    rewards = [d['r'] for d in replay_buffer]
    advantages = advs
    actions = [d['a'] for d in replay_buffer]
    
    print (np.shape(states[0]))
    print (np.shape(rewards[0]))
    print (np.shape(advantages[0]))
    print (np.shape(actions[0]))
    
    for _ in range(args.epochs):
        for batch in range(0, args.mini_batch_size, args.batch_size):
            s = batch
            e = batch + args.batch_size
            model.train(states, actions, rewards, advantages)

    _ = sess.run(set_weights)

    #####################################













