
import numpy as np
import tensorflow as tf
import cv2
import gym
import gym_ple
from collections import deque
import random

from lib.Model import Model
from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

batch_size = 32

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
        action = np.argmax(action)
        cumulated_reward = 0.0
        next_state, reward, done, _ = self.env.step(action)
        cumulated_reward += self._reward_shaping(reward)
        self.total_step += 1
        self.total_reward += reward
        return self._process(next_state), cumulated_reward, done

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
        # print (np.shape(output))
        # output = cv2.resize(output, (84, 84))
        # pretty sure this is okay bc (410 * 288 / (84 * 84 * 4)) > 1.
        output = cv2.resize(output, (80, 80))
        output = output / 255.0
        output = np.stack([output] * 4, axis=2)
        output = np.reshape(output, (1, 80, 80, 4))
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

l0 = Convolution(input_sizes=[batch_size, 80, 80, 4], filter_sizes=[8, 8, 4, 32], num_classes=None, init_filters='sqrt_fan_in', strides=[1, 4, 4, 1], padding="SAME", alpha=0., activation=Relu(), bias=0., last_layer=False, name='conv1', load=weights_conv, train=train_conv)
l1 = MaxPool(size=[batch_size, 20, 20, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
l2 = Convolution(input_sizes=[batch_size, 10, 10, 32], filter_sizes=[4, 4, 32, 64], num_classes=None, init_filters='sqrt_fan_in', strides=[1, 2, 2, 1], padding="SAME", alpha=0., activation=Relu(), bias=0., last_layer=False, name='conv2', load=weights_conv, train=train_conv)
l3 = Convolution(input_sizes=[batch_size, 5, 5, 64], filter_sizes=[3, 3, 64, 64], num_classes=None, init_filters='sqrt_fan_in', strides=[1, 1, 1, 1], padding="SAME", alpha=0., activation=Relu(), bias=0., last_layer=False, name='conv3', load=weights_conv, train=train_conv)
l4 = ConvToFullyConnected(shape=[5, 5, 64])
l5 = FullyConnected(size=[5*5*64, 512], num_classes=None, init_weights='sqrt_fan_in', alpha=0., activation=Relu(), bias=0., last_layer=False, name='fc1', load=weights_fc, train=train_fc)
l6 = FullyConnected(size=[512, 2], num_classes=None, init_weights='sqrt_fan_in', alpha=0., activation=Linear(), bias=0., last_layer=True, name='fc2', load=weights_fc, train=train_fc)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6])

####################################

predict = model.predict(state=s)
gvs = model.gvs(state=s, action=a, reward=y)
train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1.).apply_gradients(grads_and_vars=gvs)

####################################

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
    
replay_buffer = deque(maxlen=10000)

env = FlappyBirdEnv()
state = env.reset()

total_steps = 100000
for e in range(total_steps):
    print (e)
    
    #####################################

    action = predict.eval(feed_dict={s : state})
    # may need to fix reward process, bc it returns cumulative.
    next_state, reward, done = env.step(action)
    
    # we can do this much simpler and better.
    _state = np.reshape(state, (80, 80, 4))
    _action = np.reshape(action, -1)
    _reward = np.reshape(reward, -1)
    _next_state = np.reshape(next_state, (80, 80, 4))
    _done = np.reshape(done, -1)
    replay_buffer.append((_state, _action, _reward, _next_state, _done))
    
    state = next_state
    if done:
        state = env.reset()
    
    #####################################
    
    if e > 1000:
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

        y_batch = np.reshape(y_batch, -1)

        # perform gradient step
        train.run(feed_dict = {
            s : state_batch,
            a : action_batch,
            y : y_batch}
        )

    #####################################













