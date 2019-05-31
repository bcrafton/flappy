
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
    def __init__(self, sess, nbatch, nclass, epsilon, decay_max):
        self.sess = sess
        self.nbatch = nbatch
        self.nclass = nclass
        self.actions_bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.values_bias = tf.Variable(np.zeros(shape=(self.nbatch)), dtype=tf.float32)
        self.epsilon = epsilon
        self.decay_max = decay_max

        self.states = tf.placeholder("float", [None, 80, 80, 4])
        self.actions = tf.placeholder("float", [None, 2])
        self.rewards = tf.placeholder("float", [None]) 
        self.advantages = tf.placeholder("float", [None])
        
        self.actions_model1, self.values_model1 = create_model(nbatch)
        self.actions_model2, self.values_model2 = create_model(nbatch)
        
        ##############################################
        
        [self.actions1, self.actions1_forward] = self.actions_model1.forward(self.states)
        [self.actions2, self.actions2_forward] = self.actions_model2.forward(self.states)
        self.actions1_train = self.actions1 + self.actions_bias

        [self.values1, self.values1_forward] = self.values_model1.forward(self.states)
        [self.values2, self.values2_forward] = self.values_model2.forward(self.states)
        self.values1_train = self.values1 + self.values_bias
        
        ##############################################

        self.pi1 = tf.distributions.Categorical(logits=self.actions1)
        self.pi1_train = tf.distributions.Categorical(logits=self.actions1_train)
        self.pi2 = tf.distributions.Categorical(logits=self.actions2)
        
        self.opt = tf.train.AdamOptimizer(learning_rate=2.5e-4, beta1=0.9, beta2=0.999, epsilon=1.)
        self.train_op = self.opt.apply_gradients(grads_and_vars=self.gvs(self.states, self.actions, self.rewards, self.advantages))
        
        self.set_weights_actions = self.actions_model2.set_weights(self.actions_model1.get_weights())
        self.set_weights_values = self.values_model2.set_weights(self.values_model1.get_weights())
        
        self.sample_action_op = tf.squeeze(self.pi1.sample(1), axis=0, name='sample_action')
        self.eval_action = self.pi1.mode()
        
        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)

    def get_weights(self):
        assert(False)

    def set_weights(self):
        self.sess.run([self.set_weights_actions, self.set_weights_values], feed_dict={})
        self.sess.run(self.global_step_op, feed_dict={})
        
    ####################################################################
    # using other peoples code now.
    # http://blog.varunajayasiri.com/ml/ppo.html
    '''
    def policy_entropy(logits):
        a = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a = tf.exp(a)
        z = tf.reduce_sum(exp_a, axis=-1, keepdims=True)
        p = exp_a / z
        return tf.reduce_sum(p * (tf.log(z) - a), axis=-1)
    '''
    '''
    def neg_log_prob(self, action: tf.Tensor, name: str) -> tf.Tensor:
        one_hot_actions = tf.one_hot(action, 4)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pi_logits, labels=one_hot_actions, dim=-1, name=name)
    '''
    def neg_log_prob(self, action):
        # one_hot_actions = tf.one_hot(action, 2)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actions1_train, labels=action, dim=-1)
    '''
    def sample(logits):
        uniform = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(uniform)), axis=-1)
    '''
    ####################################################################

    def predict(self, state, stochastic=True):
        if stochastic:
            action_idx, value = self.sess.run([self.sample_action_op, self.values1], {self.states:[state]})
        else:
            action_idx, value = self.sess.run([self.eval_action, self.values1], {self.states:[state]})

        # reduce both to scalars.
        action_idx = np.squeeze(action_idx)
        value = np.squeeze(value)
        
        action = np.zeros(shape=2)
        action[action_idx] = 1
        
        return value, action

    def gvs(self, states, actions, rewards, advantages):
    
        ############

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)

        # ratio = tf.exp(self.pi1.log_prob(actions) - self.pi2.log_prob(actions))
        ratio = tf.exp(self.neg_log_prob(actions) - self.neg_log_prob(actions))
        
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = advantages * ratio
        surr2 = advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi1_train.entropy())

        clipped_value_estimate = self.values2 + tf.clip_by_value(self.values1_train - self.values2, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, rewards)
        value_loss_2 = tf.squared_difference(self.values1_train, rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss
        grads = tf.gradients(loss, [self.actions_bias, self.values_bias])
        [actions_grad, values_grad] = grads
        
        actions_grad = actions_grad / self.nbatch
        values_grad = values_grad / self.nbatch
        values_grad = tf.reshape(values_grad, (self.nbatch, 1))
        
        # print (actions_grad)
        # print (values_grad)
        
        action_gvs = self.actions_model1.backward(states, self.actions1_forward, actions_grad)
        values_gvs = self.values_model1.backward(states, self.values1_forward, values_grad)
        grads_and_vars = []
        grads_and_vars.extend(action_gvs)
        grads_and_vars.extend(values_gvs)
        
        return grads_and_vars

    def train(self, states, actions, rewards, advantages):
        self.train_op.run(feed_dict={self.states:states, self.actions:actions, self.rewards:rewards, self.advantages:advantages})
        
####################################
        
def create_model(nbatch):
    l1_1 = Convolution(input_sizes=[nbatch, 80, 80, 4], filter_sizes=[8, 8, 4, 32], strides=[1,4,4,1], padding="SAME", name='conv1')
    l1_2 = Relu()
    l1_3 = MaxPool(size=[nbatch, 20, 20, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = Convolution(input_sizes=[nbatch, 10, 10, 32], filter_sizes=[4, 4, 32, 64], strides=[1,2,2,1], padding="SAME", name='conv2')
    l2_2 = Relu()

    l3_1 = Convolution(input_sizes=[nbatch, 5, 5, 64], filter_sizes=[3, 3, 64, 64], strides=[1,1,1,1], padding="SAME", name='conv3')
    l3_2 = Relu()

    l4 = ConvToFullyConnected(input_shape=[5, 5, 64])

    l5_1 = FullyConnected(input_shape=5*5*64, size=512, name='fc1')
    l5_2 = Relu()

    actions = FullyConnected(input_shape=512, size=2, name='action')
    values = FullyConnected(input_shape=512, size=1, name='values')

    actions_model = Model(layers=[l1_1, l1_2, l1_3, \
                                  l2_1, l2_2,       \
                                  l3_1, l3_2,       \
                                  l4,               \
                                  l5_1, l5_2,       \
                                  actions           \
                                  ])

    values_model = Model(layers=[l1_1, l1_2, l1_3, \
                                 l2_1, l2_2,       \
                                 l3_1, l3_2,       \
                                 l4,               \
                                 l5_1, l5_2,       \
                                 values            \
                                 ])

    return actions_model, values_model

####################################
        
        
        
        
