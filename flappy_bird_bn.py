
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
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

batch_size = 64
epsilon = 0.05
decay = 0.99
total_steps = int(1e6)

####################################

class FlappyBirdEnv:
    def __init__(self):
        self.env = gym.make('FlappyBird-v0')
        self.env.seed(np.random.randint(0, 100000))
        self.total_reward = 0.0
        self.total_step = 0

    def reset(self):
        state = self.env.reset()
        self.total_reward = 0.0
        self.total_step = 0
        return self._process(state)

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        reward = self._reward_shaping(reward)
        self.total_step += 1
        self.total_reward += reward
        return self._process(next_state), reward, done

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
        output = np.stack([output] * 4, axis=2)
        return output

####################################

train_fc = True
weights_fc = None

train_conv = True
weights_conv = None

####################################

s = tf.placeholder("float", [None, 80, 80, 4])
a = tf.placeholder("float", [None, 2])
y = tf.placeholder("float", [None])

l1_1 = Convolution(input_sizes=[batch_size, 80, 80, 4], filter_sizes=[8, 8, 4, 32], init='alexnet', strides=[1,4,4,1], padding="SAME", name='conv1', load=weights_conv)
l1_2 = BatchNorm(input_size=[batch_size, 20, 20, 32], name='conv1_bn', load=weights_conv)
l1_3 = Relu()
l1_4 = MaxPool(size=[batch_size, 20, 20, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = Convolution(input_sizes=[batch_size, 10, 10, 32], filter_sizes=[4, 4, 32, 64], init='alexnet', strides=[1,2,2,1], padding="SAME", name='conv2', load=weights_conv)
l2_2 = BatchNorm(input_size=[batch_size, 5, 5, 64], name='conv2_bn', load=weights_conv)
l2_3 = Relu()

l3_1 = Convolution(input_sizes=[batch_size, 5, 5, 64], filter_sizes=[3, 3, 64, 64], init='alexnet', strides=[1,1,1,1], padding="SAME", name='conv3', load=weights_conv)
l3_2 = BatchNorm(input_size=[batch_size, 5, 5, 64], name='conv3_bn', load=weights_conv)
l3_3 = Relu()

l4 = ConvToFullyConnected(input_shape=[5, 5, 64])

l5_1 = FullyConnected(input_shape=5*5*64, size=512, init='alexnet', name='fc1', load=weights_fc)
l5_2 = BatchNorm(input_size=[batch_size, 512], name='fc1_bn', load=weights_fc)
l5_3 = Relu()

l6 = FullyConnected(input_shape=512, size=2, init='alexnet', name='fc2', load=weights_fc)

model = Model(layers=[l1_1, l1_2, l1_3, l1_4, \
                      l2_1, l2_2, l2_3,       \
                      l3_1, l3_2, l3_3,       \
                      l4,                     \
                      l5_1, l5_2, l5_3,       \
                      l6,                     \
                      ])

####################################

predict = model.predict(state=s)
gvs = model.gvs(state=s, action=a, reward=y)
get_weights = model.get_weights()
train = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=gvs)

####################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

####################################

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
    
replay_buffer = deque(maxlen=10000)

env = FlappyBirdEnv()
state = env.reset()

action_list = []
for e in range(total_steps):
    
    #####################################

    # should decay epsilon here ...
    # rand < eps because eps is % random actions. 
    if np.random.rand() < epsilon:
        action_idx = env.env.action_space.sample()
    else:
        q_value = predict.eval(feed_dict={s : [state]})
        action_idx = np.argmax(q_value)

    action = np.zeros(2)
    action[action_idx] = 1
    action_list.append(action_idx)
    
    next_state, reward, done = env.step(action_idx)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state
    
    if done:
        p = "%d %f" % (e, env.total_reward)
        print (p)
        f = open(filename, "a")
        f.write(p + "\n")
        f.close()

        action_list = []
        state = env.reset()
    
    #####################################

    if e > 1000:
        # could be far more efficient here if we stored the processed state, action, y values somewhere else.
        minibatch = random.sample(replay_buffer, batch_size)

        # get the batch variables
        state_batch = [d[0] for d in minibatch]
        action_batch = [d[1] for d in minibatch]
        reward_batch = [d[2] for d in minibatch]
        next_state_batch = [d[3] for d in minibatch]

        y_batch = []
        next_reward_batch = predict.eval(feed_dict={s : next_state_batch})
        for i in range(0, len(minibatch)):
            done = minibatch[i][4]
            
            # if done, only equals reward
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + 0.99 * np.max(next_reward_batch[i]))

        # perform gradient step
        train.run(feed_dict = {s:state_batch, a:action_batch, y:y_batch})

    if (e % 1000) == 0:
        w = sess.run(get_weights)
        np.save('weights', w)

    #####################################













