
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
        self.use_tf = True

        self.sess = sess
        self.nbatch = nbatch
        self.nclass = nclass
        self.logits_bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.values_bias = tf.Variable(np.zeros(shape=(self.nbatch)), dtype=tf.float32)
        self.epsilon = epsilon
        self.decay_max = decay_max

        self.states = tf.placeholder("float", [None, 84, 84, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])

        if self.use_tf:
            self.logits, self.pi, self.values, self.params = self.create_model_tf()
        else:
            self.actions_model, self.values_model = self.create_model(nbatch)

        ##############################################

        if not self.use_tf:
            [self.logits, self.logits_forward] = self.actions_model.forward(self.states)
            [self.values, self.values_forward] = self.values_model.forward(self.states)

        self.logits_train = self.logits + self.logits_bias
        self.values_train = self.values + self.values_bias

        ##############################################

        self.pi1 = tf.distributions.Categorical(logits=self.logits)
        self.pi2 = tf.distributions.Categorical(logits=self.logits_train)

        self.actions        = tf.squeeze(self.pi1.sample(1), axis=0)
        
        self.nlps1          = self.pi1.log_prob(self.actions)
        self.nlps2          = self.pi2.log_prob(self.old_actions)

        self.policy_entropy = policy_entropy(self.logits_train)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)

        ##############################################

        ratio = tf.exp(self.nlps2 - self.old_nlps)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi2.entropy())

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.values_train - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.values_train, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        self.loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss

        ##############################################

        self.opt = tf.train.AdamOptimizer(learning_rate=2.5e-4, epsilon=1e-5)

        if self.use_tf:
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), 0.5)
            grads_and_vars = list(zip(grads, self.params))
            self.train_op = self.opt.apply_gradients(grads_and_vars)
        else:
            self.train_op = self.opt.apply_gradients(grads_and_vars=self.gvs(self.states, self.rewards, self.advantages, self.old_actions, self.old_values, self.old_nlps))

        # self.grads_op = self.gvs(self.states, self.rewards, self.advantages, self.old_actions, self.old_values, self.old_nlps)

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

    def gvs(self, states, rewards, advantages, old_actions, old_values, old_nlps):

        grads = tf.gradients(self.loss, [self.logits_bias, self.values_bias])
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        [logits_grad, values_grad] = grads

        # logits_grad = tf.clip_by_global_norm(logits_grad, 0.5)     
        # logits_grad = logits_grad / self.nbatch
        # values_grad = tf.clip_by_global_norm(values_grad, 0.5) 
        # values_grad = values_grad / self.nbatch
        values_grad = tf.reshape(values_grad, (self.nbatch, 1))
        
        logits_gvs = self.actions_model.backward(states, self.logits_forward, logits_grad)
        values_gvs = self.values_model.backward(states, self.values_forward, values_grad)
        grads_and_vars = []
        grads_and_vars.extend(logits_gvs)
        grads_and_vars.extend(values_gvs)
        
        return grads_and_vars

    def train(self, states, rewards, advantages, old_actions, old_values, old_nlps):
        self.train_op.run(feed_dict={self.states:states, 
                                     self.rewards:rewards, 
                                     self.advantages:advantages, 
                                     self.old_actions:old_actions, 
                                     self.old_values:old_values, 
                                     self.old_nlps:old_nlps})

    def grads(self, states, rewards, advantages, old_actions, old_values, old_nlps):
        ret = self.sess.run(self.grads_op, 
                            feed_dict={self.states:states, 
                                       self.rewards:rewards, 
                                       self.advantages:advantages, 
                                       self.old_actions:old_actions, 
                                       self.old_values:old_values, 
                                       self.old_nlps:old_nlps})
        return ret

    
    def create_model_tf(self):        
        conv1 = tf.layers.conv2d(self.states, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)
        flattened = tf.layers.flatten(conv3)
        fc = tf.layers.dense(flattened, 512, activation=tf.nn.relu)
        
        action_logits = tf.layers.dense(fc, 4)
        action_dists = tf.distributions.Categorical(logits=action_logits)
        values = tf.squeeze(tf.layers.dense(fc, 1), axis=-1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    
        return action_logits, action_dists, values, params

    def create_model(self, nbatch):
        l1_1 = Convolution(input_sizes=[nbatch, 84, 84, 4], filter_sizes=[8, 8, 4, 32], strides=[1,4,4,1], init='alexnet', padding="VALID", name='conv1')
        l1_2 = Relu()

        l2_1 = Convolution(input_sizes=[nbatch, 20, 20, 32], filter_sizes=[4, 4, 32, 64], strides=[1,2,2,1], init='alexnet', padding="VALID", name='conv2')
        l2_2 = Relu()

        l3_1 = Convolution(input_sizes=[nbatch, 9, 9, 64], filter_sizes=[3, 3, 64, 64], strides=[1,1,1,1], init='alexnet', padding="VALID", name='conv3')
        l3_2 = Relu()

        l4 = ConvToFullyConnected(input_shape=[7, 7, 64])

        l5_1 = FullyConnected(input_shape=7*7*64, size=512, init='alexnet', name='fc1')
        l5_2 = Relu()

        actions = FullyConnected(input_shape=512, size=4, init='alexnet', name='action')
        values = FullyConnected(input_shape=512, size=1, init='alexnet', name='values')

        actions_model = Model(layers=[l1_1, l1_2,       \
                                      l2_1, l2_2,       \
                                      l3_1, l3_2,       \
                                      l4,               \
                                      l5_1, l5_2,       \
                                      actions           \
                                      ])

        values_model = Model(layers=[l1_1, l1_2,       \
                                     l2_1, l2_2,       \
                                     l3_1, l3_2,       \
                                     l4,               \
                                     l5_1, l5_2,       \
                                     values            \
                                     ])

        return actions_model, values_model


####################################
        
        
        
        
