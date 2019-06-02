
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

def sample(logits):
    uniform = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(uniform)), axis=-1)

def policy_entropy(logits):
    a = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    exp_a = tf.exp(a)
    z = tf.reduce_sum(exp_a, axis=-1, keepdims=True)
    p = exp_a / z
    return tf.reduce_sum(p * (tf.log(z) - a), axis=-1)

def neg_log_prob(logits, actions):
    one_hot_actions = tf.one_hot(actions, 4)
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions, dim=-1)

class PPOModel:
    def __init__(self, sess, nbatch, nclass, epsilon, decay_max):
        self.sess = sess
        self.nbatch = nbatch
        self.nclass = nclass
        self.epsilon = epsilon
        self.decay_max = decay_max

        self.states = tf.placeholder("float", [None, 84, 84, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])

        self.logits, self.values, self.params = self.create_model()
        
        ##############################################

        self.actions        = sample(self.logits)
        self.policy_entropy = policy_entropy(self.logits)
        self.nlps           = neg_log_prob(logits=self.logits, actions=self.actions)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)
        
        ############

        ratio = tf.exp(self.old_nlps - self.nlps)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon_decay, 1.0 + epsilon_decay, name="clipped_ratio")
        policy_reward = tf.reduce_mean(tf.minimum(ratio * self.advantages, clipped_ratio * self.advantages), name="policy_reward")

        entropy_bonus = tf.reduce_mean(self.policy_entropy)

        clipped_value = tf.add(self.old_values, tf.clip_by_value(self.values - self.old_values, -epsilon_decay, epsilon_decay))
        vf_loss = tf.multiply(0.5, tf.reduce_mean(tf.maximum(tf.square(self.values - self.rewards), tf.square(clipped_value - self.rewards))), name="vf_loss")
        loss = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        ##############################################

        self.opt = tf.train.AdamOptimizer(learning_rate=2.5e-4, beta1=0.9, beta2=0.999, epsilon=1.)
        self.train_op = self.opt.minimize(loss, var_list=self.params)

        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)
        
    def get_weights(self):
        assert(False)

    def set_weights(self):
        self.sess.run(self.global_step_op, feed_dict={})

    ####################################################################

    def predict(self, state):
        action, value, nlp = self.sess.run([self.actions, self.values, self.nlps], {self.states:[state]})

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
        
        action_logits = tf.layers.dense(fc, 4)
        values = tf.squeeze(tf.layers.dense(fc, 1), axis=-1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    
        return action_logits, values, params

####################################
        
        
        
        
