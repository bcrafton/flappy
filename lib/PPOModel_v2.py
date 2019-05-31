
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
        self.logits_bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.values_bias = tf.Variable(np.zeros(shape=(self.nbatch)), dtype=tf.float32)
        self.epsilon = epsilon
        self.decay_max = decay_max

        self.states = tf.placeholder("float", [None, 80, 80, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])

        self.actions_model, self.values_model = create_model(nbatch)
        
        ##############################################

        [self.logits, self.logits_forward] = self.actions_model.forward(self.states)
        self.logits_train = self.logits + self.logits_bias

        [self.values, self.values_forward] = self.values_model.forward(self.states)
        self.values_train = self.values + self.values_bias
        
        ##############################################

        self.actions        = sample(self.logits)
        self.policy_entropy = policy_entropy(self.logits_train)
        self.nlps           = neg_log_prob(logits=self.logits, actions=self.actions)
        self.nlps_train     = neg_log_prob(logits=self.logits_train, actions=self.old_actions)

        self.opt = tf.train.AdamOptimizer(learning_rate=2.5e-4, beta1=0.9, beta2=0.999, epsilon=1.)
        self.train_op = self.opt.apply_gradients(grads_and_vars=self.gvs(self.states, self.rewards, self.advantages, self.old_actions, self.old_values, self.old_nlps))

        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)
        
    def get_weights(self):
        assert(False)

    def set_weights(self):
        self.sess.run(self.global_step_op, feed_dict={})

    ####################################################################

    def predict(self, state):
        value, action, nlp = self.sess.run([self.values, self.actions, self.nlps], {self.states:[state]})

        action = np.squeeze(action)
        value = np.squeeze(value)
        nlp = np.squeeze(nlp)
        
        return action, value, nlp

    def gvs(self, states, rewards, advantages, old_actions, old_values, old_nlps):
    
        ############

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)
        
        ############

        ratio = tf.exp(old_nlps - self.nlps_train)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon_decay, 1.0 + epsilon_decay, name="clipped_ratio")
        policy_reward = tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages), name="policy_reward")

        entropy_bonus = tf.reduce_mean(self.policy_entropy)

        clipped_value = tf.add(old_values, tf.clip_by_value(self.values_train - old_values, -epsilon_decay, epsilon_decay))
        vf_loss = tf.multiply(0.5, tf.reduce_mean(tf.maximum(tf.square(self.values_train - rewards), tf.square(clipped_value - rewards))), name="vf_loss")
        loss = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        ############
        
        grads = tf.gradients(loss, [self.logits_bias, self.values_bias])
        [logits_grad, values_grad] = grads
        
        logits_grad = logits_grad / self.nbatch
        values_grad = values_grad / self.nbatch
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

    actions = FullyConnected(input_shape=512, size=4, name='action')
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
        
        
        
        
