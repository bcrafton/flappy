
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
        self.frame_skip = 4

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

train_fc = True
weights_fc = None

train_conv = True
weights_conv = None

####################################

def create_model():
    s = tf.placeholder("float", [None, 80, 80, 4])
    a = tf.placeholder("float", [None, 2])
    y = tf.placeholder("float", [None])
    adv = tf.placeholder("float", [None])

    l1_1 = Convolution(input_sizes=[args.batch_size, 80, 80, 4], filter_sizes=[8, 8, 4, 32], init='alexnet', strides=[1,4,4,1], padding="SAME", name='conv1', load=weights_conv)
    l1_2 = BatchNorm(input_size=[args.batch_size, 20, 20, 32], name='conv1_bn', load=weights_conv)
    l1_3 = Relu()
    l1_4 = MaxPool(size=[args.batch_size, 20, 20, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = Convolution(input_sizes=[args.batch_size, 10, 10, 32], filter_sizes=[4, 4, 32, 64], init='alexnet', strides=[1,2,2,1], padding="SAME", name='conv2', load=weights_conv)
    l2_2 = BatchNorm(input_size=[args.batch_size, 5, 5, 64], name='conv2_bn', load=weights_conv)
    l2_3 = Relu()

    l3_1 = Convolution(input_sizes=[args.batch_size, 5, 5, 64], filter_sizes=[3, 3, 64, 64], init='alexnet', strides=[1,1,1,1], padding="SAME", name='conv3', load=weights_conv)
    l3_2 = BatchNorm(input_size=[args.batch_size, 5, 5, 64], name='conv3_bn', load=weights_conv)
    l3_3 = Relu()

    l4 = ConvToFullyConnected(input_shape=[5, 5, 64])

    l5_1 = FullyConnected(input_shape=5*5*64, size=512, init='alexnet', name='fc1', load=weights_fc)
    l5_2 = BatchNorm(input_size=[args.batch_size, 512], name='fc1_bn', load=weights_fc)
    l5_3 = Relu()

    l6 = FullyConnected(input_shape=512, size=2, init='alexnet', name='fc2', load=weights_fc)

    model = Model(layers=[l1_1, l1_2, l1_3, l1_4, \
                          l2_1, l2_2, l2_3,       \
                          l3_1, l3_2, l3_3,       \
                          l4,                     \
                          l5_1, l5_3,             \
                          # l5_1, l5_2, l5_3,       \
                          l6,                     \
                          ])

    predict = model.predict(state=s)
    get_weights = model.get_weights()
    
    gvs = model.gvs(state=s, action=a, reward=y)
    train = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=gvs)

    return s, a, y, adv, model, predict, get_weights, train 

####################################

s1, a1, y1, adv1, model1, predict1, get_weights1, train1 = create_model()
s2, a2, y2, adv2, model2, predict2, get_weights2, train2 = create_model()
set_weights = model2.set_weights(get_weights1)

####################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model1.num_params()) + "\n")
f.close()

####################################

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

####################################

train_data = deque(maxlen=10000)
replay_buffer = []

env = FlappyBirdEnv()
state = env.reset()

for e in range(total_episodes):
    
    epsilon = epsilon_init - e * decay_rate
    
    #####################################

    replay_buffer = []
    for _ in range(args.mini_batch_size):

        q_value = predict1.eval(feed_dict={s1 : [state]})
        q_value = np.squeeze(q_value)
        
        if np.random.rand() < epsilon:
            action_idx = env.env.action_space.sample()
        else:
            action_idx = np.argmax(q_value)
            
        value = q_value[action_idx]
        
        ################################

        action = np.zeros(2)
        action[action_idx] = 1
        next_state, reward, done = env.step(action_idx)

        if done and env.total_step >= 10000:
            next_q_value = predict1.eval(feed_dict={s1 : [next_state]})
            next_value = np.max(next_q_value)
            reward += 0.99 * next_value
        
        replay_buffer.append({'s':state, 'v': value, 'a':action, 'r':reward, 'd':done})
        state = next_state
        
        if done:
            state = env.reset()

    next_q_value = predict1.eval(feed_dict={s1 : [next_state]})
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
            train1.run(feed_dict = {s1:states[s:e], a1:actions[s:e], y1:rets[s:e], adv1:advs[s:e]})

    _ = sess.run(set_weights)

    #####################################













