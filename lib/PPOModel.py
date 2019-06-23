
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

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

class PPOModel:
    def __init__(self, sess, nbatch, nclass, epsilon, decay_max, lr=2.5e-4, eps=1e-2):

        self.sess = sess
        self.nbatch = nbatch
        self.nclass = nclass
        self.logits_bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.values_bias = tf.Variable(np.zeros(shape=(self.nbatch, 1)), dtype=tf.float32)
        self.epsilon = epsilon
        self.decay_max = decay_max
        self.lr = lr
        self.eps = eps

        self.states = tf.placeholder("float", [None, 84, 84, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])

        self.values, self.pi, self.params = self.create_model()
        self.values = tf.reshape(self.values, (-1,))
        
        ##############################################

        self.actions = tf.squeeze(self.pi.sample(1), axis=0)        
        self.nlps1 = self.pi.log_prob(self.actions)
        self.nlps2 = self.pi.log_prob(self.old_actions)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)

        ##############################################

        ratio = tf.exp(self.nlps2 - self.old_nlps)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi.entropy())

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.values - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.values, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        self.loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss

        ##############################################

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps)
        self.train_op = self.opt.minimize(self.loss)

        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)
        
    def get_weights(self):
        assert(False)

    def set_weights(self):
        self.sess.run(self.global_step_op, feed_dict={})

    ####################################################################

    def predict(self, state):
        action, value, nlp = self.sess.run([self.actions, self.values, self.nlps1], {self.states:[state]})

        action = np.squeeze(action)
        value = np.squeeze(value)
        nlp = np.squeeze(nlp)
        
        return action, value, nlp

    def train(self, states, rewards, advantages, old_actions, old_values, old_nlps):
        self.train_op.run(feed_dict={self.states:states, 
                                     self.rewards:rewards, 
                                     self.advantages:advantages, 
                                     self.old_actions:old_actions, 
                                     self.old_values:old_values, 
                                     self.old_nlps:old_nlps})

    def create_model(self):
        conv1 = tf.layers.conv2d(self.states, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)
        flattened = tf.layers.flatten(conv3)
        fc = tf.layers.dense(flattened, 512, activation=tf.nn.relu)

        values = tf.squeeze(tf.layers.dense(fc, 1), axis=-1)
        action_logits = tf.layers.dense(fc, 4)
        action_dists = tf.distributions.Categorical(logits=action_logits)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        
        return values, action_dists, params

    ####################################################################
        
        
        
        
