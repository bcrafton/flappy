
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
        
        l1 = AvgPool(size=self.input_shape, ksize=self.pool_shape, strides=self.pool_shape, padding='SAME')

        l2_input_shape = l1.output_shape()
        l2 = ConvToFullyConnected(input_shape=l2_input_shape)
        
        l3_input_shape = l2.output_shape()
        actions = FullyConnected(input_shape=l3_input_shape, size=self.nactions, init='alexnet', name=self.name + '_actions')
        values = FullyConnected(input_shape=l3_input_shape, size=1, init='alexnet', name=self.name + '_values')

        ####################################################

        self.logits_bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.values_bias = tf.Variable(np.zeros(shape=(self.nbatch, 1)), dtype=tf.float32)
        
        self.actions_model = Model(layers=[l1, l2, actions])
        self.values_model = Model(layers=[l1, l2, values])

        ####################################################

        # we cant have a placeholder for this.
        self.states = tf.placeholder("float", [None, 84, 84, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])
        
        ####################################################
        
        [self.logits, self.logits_forward] = self.actions_model.forward(self.states)
        [self.values, self.values_forward] = self.values_model.forward(self.states)
        
        self.logits_train = self.logits + self.logits_bias
        self.values_train = self.values + self.values_bias

        self.values       = tf.reshape(self.values,       (-1,))
        self.values_train = tf.reshape(self.values_train, (-1,))
        
        ####################################################

        # gonna have to get rid of pi i think ... 
        # go to the old sample, neg_log_prob thing. 
        # need to make sure equivalent in a test script. 
        self.pi1 = tf.distributions.Categorical(logits=self.logits)
        self.pi2 = tf.distributions.Categorical(logits=self.logits_train)

        self.actions        = tf.squeeze(self.pi1.sample(1), axis=0)
        
        self.nlps1          = self.pi1.log_prob(self.actions)
        self.nlps2          = self.pi2.log_prob(self.old_actions)

    ###################################################################
    
    def get_weights(self):
        return []
        
    def output_shape(self):
        return self.input_shape

    def num_params(self):
        return 0
        
    def forward(self, X):
        return X
        
    def predict(self, X):
        # we are gonna need to move stuff around the constructor. 
        
        # states, rewards, advantages, old_actions, old_values, old_nlps
        cache = {self.action_name: self.actions, self.value_name: self.values, self.nlp_name: self.nlps1}
        return X, cache
                
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return DO

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return DO
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
        
    def lel_backward(self, AI, AO, E, DO, Y):
        # write this
        
        
    def lel_gv(self, AI, AO, E, DO, Y):
        # write this
        assert(False)

    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
        
        


