
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

        self.actions_model, self.values_model, self.headless, self.actions_head, self.values_head = self.create_model(nbatch)
        weights1 = self.actions_model.get_weights()
        weights2 = self.values_model.get_weights()
        weights = {}
        weights.update(weights1)
        weights.update(weights2)
        self.params = []
        for key in weights.keys():
            self.params.append(weights[key])

        ##############################################

        [self.logits, self.logits_forward] = self.actions_model.forward(self.states)
        [self.values, self.values_forward] = self.values_model.forward(self.states)

        self.logits_train = self.logits + self.logits_bias
        self.values_train = self.values + self.values_bias
        
        self.values       = tf.reshape(self.values,       (-1,))
        self.values_train = tf.reshape(self.values_train, (-1,))

        ##############################################

        self.pi1 = tf.distributions.Categorical(logits=self.logits)
        self.pi2 = tf.distributions.Categorical(logits=self.logits_train)

        self.actions        = tf.squeeze(self.pi1.sample(1), axis=0)
        
        self.nlps1          = self.pi1.log_prob(self.actions)
        self.nlps2          = self.pi2.log_prob(self.old_actions)

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

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps)
        self.train_op = self.opt.apply_gradients(grads_and_vars=self.gvs(self.states, self.rewards, self.advantages, self.old_actions, self.old_values, self.old_nlps))

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

        grads = tf.gradients(self.loss, [self.logits_bias, self.values_bias] + self.params)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)

        logits_grad = grads[0]
        values_grad = grads[1]

        logits_back = self.actions_head.backward(self.logits_forward[-2], self.logits_forward[-1], logits_grad)
        logits_gvs  = self.actions_head.gv(self.logits_forward[-2], self.logits_forward[-1], logits_grad)
 
        values_back = self.values_head.backward(self.values_forward[-2], self.values_forward[-1], values_grad)
        values_gvs  = self.values_head.gv(self.values_forward[-2], self.values_forward[-1], values_grad)

        gvs = self.headless.backward(states, self.logits_forward, logits_back + values_back)

        grads_and_vars = []
        grads_and_vars.extend(logits_gvs)
        grads_and_vars.extend(values_gvs)
        grads_and_vars.extend(gvs)

        return grads_and_vars

    def train(self, states, rewards, advantages, old_actions, old_values, old_nlps):
        self.train_op.run(feed_dict={self.states:states, 
                                     self.rewards:rewards, 
                                     self.advantages:advantages, 
                                     self.old_actions:old_actions, 
                                     self.old_values:old_values, 
                                     self.old_nlps:old_nlps})

    def create_model(self, nbatch):
        l1_1 = Convolution(input_sizes=[nbatch, 84, 84, 4], filter_sizes=[8, 8, 4, 32], strides=[1,4,4,1], padding="VALID", name='conv1')
        l1_2 = Relu()

        l2_1 = Convolution(input_sizes=[nbatch, 20, 20, 32], filter_sizes=[4, 4, 32, 64], strides=[1,2,2,1], padding="VALID", name='conv2')
        l2_2 = Relu()

        l3_1 = Convolution(input_sizes=[nbatch, 9, 9, 64], filter_sizes=[3, 3, 64, 64], strides=[1,1,1,1], padding="VALID", name='conv3')
        l3_2 = Relu()

        l4 = ConvToFullyConnected(input_shape=[7, 7, 64])

        l5_1 = FullyConnected(input_shape=7*7*64, size=512, name='fc1')
        l5_2 = Relu()

        actions = FullyConnected(input_shape=512, size=4, name='action')
        values = FullyConnected(input_shape=512, size=1, name='values')

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

        headless = Model(layers=[l1_1, l1_2,       \
                                 l2_1, l2_2,       \
                                 l3_1, l3_2,       \
                                 l4,               \
                                 l5_1, l5_2        \
                                 ])

        return actions_model, values_model, headless, actions, values


    ####################################################################
        
        
        
        
