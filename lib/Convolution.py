
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

class Convolution(Layer):

    def __init__(self, input_sizes, filter_sizes, strides, padding, init=None, alpha=0., activation=None, bias=0., name=None, load=None, train=True):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        
        bias = np.ones(shape=self.fout) * bias
        
        self.strides = strides
        self.padding = padding
        self.alpha = alpha
        self.activation = Linear() if activation == None else activation
        self.name = name
        self._train = train
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1').item()
            filters = weight_dict[self.name]
            bias = weight_dict[self.name + '_bias']
        else:
            if init == "zero":
                filters = np.zeros(shape=self.filter_sizes)
            elif init == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
                filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.filter_sizes)
            elif init == "alexnet":
                filters = np.random.normal(loc=0.0, scale=0.01, size=self.filter_sizes)
            else:
                # https://www.tensorflow.org/api_docs/python/tf/glorot_uniform_initializer
                # can verify we did this right ...
                fan_in = self.fh * self.fw * self.fin
                fan_out = self.fout
                lim = np.sqrt(6. / (fan_in + fan_out))
                filters = np.random.uniform(low=-lim, high=lim, size=self.filter_sizes)
                
        self.filters = tf.Variable(filters, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def set_weights(self, weight_dic):
        filters = weight_dic[self.name]
        bias = weight_dic[self.name + '_bias']
        return [self.filters.assign(filters), self.bias.assign(bias)]

    def output_shape(self):
        oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        od = self.fout
        return [oh, oh, od]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        Z = tf.nn.conv2d(X, self.filters, self.strides, self.padding) + tf.reshape(self.bias, [1, 1, self.fout])
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def gv(self, AI, AO, DO):    
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################    
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return self.backward(AI, AO, DO)

    def lel_gv(self, AI, AO, E, DO, Y):
        return self.gv(AI, AO, DO)
        
    def lel(self, AI, AO, E, DO, Y): 
        return self.train(AI, AO, DO)
    ################################################################### 
        
        
