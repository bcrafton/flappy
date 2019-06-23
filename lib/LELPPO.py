
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.FeedbackMatrix import FeedbackMatrix

from lib.Model import Model
from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Activation import Relu
from lib.Activation import Linear

# so we actually dont need ppo_cache in :
# def lel_backward(self, AI, AO, DO, ppo_cache):
# bc we shud be just get it when sess.run is called. 
# i think its better to call sess.run once ...
# cud do either way tho...

# where every LELPPO layer pulls the accumulated things it said to cache in predict. 
# and calls sess.run to create a new DO.
# but we wud already be running a session so nah ...

####################################################

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
    one_hot_actions = tf.one_hot(actions, action_dim)
    return -tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions, dim=-1)
    
####################################################

class LELPPO(Layer):

    def __init__(self, input_shape, pool_shape, nactions, name=None):
        self.input_shape = input_shape
        self.batch_size, self.h, self.w, self.fin = self.input_shape
        self.pool_shape = pool_shape
        self.nactions = nactions
        self.name = name
        self.action_name = self.name + '_action'
        self.value_name = self.name + '_value'
        self.nlp_name = self.name + '_nlp'
        
        self.pool = AvgPool(size=self.input_shape, ksize=self.pool_shape, strides=self.pool_shape, padding='SAME')

        l2_input_shape = l1.output_shape()
        self.conv2fc = ConvToFullyConnected(input_shape=l2_input_shape)
        
        l3_input_shape = l2.output_shape()
        self.actions = FullyConnected(input_shape=l3_input_shape, size=self.nactions, init='alexnet', name=self.name + '_actions')
        self.values = FullyConnected(input_shape=l3_input_shape, size=1, init='alexnet', name=self.name + '_values')

        ####################################################

        self.logits_bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.values_bias = tf.Variable(np.zeros(shape=(self.nbatch, 1)), dtype=tf.float32)
        
        # self.actions_model = Model(layers=[l1, l2, actions])
        # self.values_model = Model(layers=[l1, l2, values])

        ####################################################

        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])
        
        ####################################################
        
    def get_weights(self):
        return []
        
    def output_shape(self):
        return self.input_shape

    def num_params(self):
        return 0
        
    def place_holders(self):
        place_holders_dict = {}
        place_holders_dict[self.name + '_advantages'] = self.advantages
        place_holders_dict[self.name + '_rewards'] = self.rewards
        place_holders_dict[self.name + '_old_actions'] = self.old_actions
        place_holders_dict[self.name + '_old_values'] = self.old_values
        place_holders_dict[self.name + '_old_nlps'] = self.old_nlps
        return place_holders_dict
        
    ###################################################################
        
    def forward(self, X):
        return X
        
    def predict(self, X):
        # [logits, logits_forward] = self.actions_model.forward(X) 
        # [values, values_forward] = self.values_model.forward(X)
        
        pool = self.pool.forward(AI)
        conv2fc = self.conv2fc.forward(pool)
        logits = self.actions.forward(conv2fc)
        values = self.values.forward(conv2fc)
        
        values = tf.reshape(values, (-1,))
        actions = sample(logits)
        nlps = neg_log_prob(logits, actions) 

        # states, rewards, advantages, old_actions, old_values, old_nlps
        cache = {self.action_name: actions, self.value_name: values, self.nlp_name: nlps}
        return X, cache
                
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return DO

    def gv(self, AI, AO, DO):    
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return DO
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    ###################################################################   
        
    def lel_backward(self, AI, AO, DO, cache):
    
        pool = self.pool.forward(AI)
        conv2fc = self.conv2fc.forward(pool)
        logits = self.actions.forward(conv2fc)
        values = self.values.forward(conv2fc)
    
        # [logits, logits_forward] = self.actions_model.forward(AI)
        # [values, values_forward] = self.values_model.forward(AI)
        
        logits = logits + self.logits_bias
        values = values + self.values_bias
        values = tf.reshape(values, (-1,))
        nlps = neg_log_prob(logits, self.old_actions)
        
        ratio = tf.exp(nlps - self.old_nlps)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -policy_entropy(train)

        clipped_value_estimate = self.old_values + tf.clip_by_value(values - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(values, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        
        ###################################################################

        loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss
        # grads = tf.gradients(self.loss, [self.logits_bias, self.values_bias] + self.params)
        grads = tf.gradients(self.loss, [self.logits_bias, self.values_bias])
        
        do_logits = grads[0]
        do_values = grads[1]
        
        # we never call forward in lel, until backwards... forward just returns X.
        # actually works out nicely.
        # perhaps we dont actually need a cache then. 
        # a few cheap redundant computations isnt so bad.

        dlogits = self.actions.backward(conv2fc, logits, do_logits)
        dvalues = self.values.backward(conv2fc, values, do_values)
        dconv2fc = self.conv2fc.backward(pool, conv2fc, dlogits + dvalues)
        dpool = self.pool.backward(AI, pool, dconv2fc)
        
        return dpool
        
    def lel_gv(self, AI, AO, DO, cache):
    
        pool = self.pool.forward(AI)
        conv2fc = self.conv2fc.forward(pool)
        logits = self.actions.forward(conv2fc)
        values = self.values.forward(conv2fc)
    
        # [logits, logits_forward] = self.actions_model.forward(AI)
        # [values, values_forward] = self.values_model.forward(AI)
        
        logits = logits + self.logits_bias
        values = values + self.values_bias
        values = tf.reshape(values, (-1,))
        nlps = neg_log_prob(logits, self.old_actions)
        
        ratio = tf.exp(nlps - self.old_nlps)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -policy_entropy(train)

        clipped_value_estimate = self.old_values + tf.clip_by_value(values - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(values, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        
        ###################################################################

        loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss
        # grads = tf.gradients(self.loss, [self.logits_bias, self.values_bias] + self.params)
        grads = tf.gradients(self.loss, [self.logits_bias, self.values_bias])
        
        do_logits = grads[0]
        do_values = grads[1]
        
        # we never call forward in lel, until backwards... forward just returns X.
        # actually works out nicely.
        # perhaps we dont actually need a cache then. 
        # a few cheap redundant computations isnt so bad.

        gvs = []
        dlogits = self.actions.gv(conv2fc, logits, do_logits)
        dvalues = self.values.gv(conv2fc, values, do_values)
        # dconv2fc = self.conv2fc.backward(pool, conv2fc, dlogits + dvalues)
        # dpool = self.pool.backward(AI, pool, dconv2fc)
        
        gvs.extend(dlogits, dvalues)
        
        return gvs
        
    ###################################################################
        
        


