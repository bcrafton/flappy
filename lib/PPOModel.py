
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
    def __init__(self, nbatch, nclass, epsilon, decay):
        self.nbatch = nbatch
        self.nclass = nclass
        self.bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        
        # want to move random action in here.
        self.epsilon = epsilon
        self.decay = decay

        self.states = tf.placeholder("float", [None, 80, 80, 4])
        self.actions = tf.placeholder("float", [None, 2])
        self.rewards = tf.placeholder("float", [None]) 
        self.advantages = tf.placeholder("float", [None])
        
        self.model1 = create_model(nbatch)
        self.model2 = create_model(nbatch)
        
        self.predict_op1 = self.model1.predict(self.states)
        self.predict_op2 = self.model2.predict(self.states)
        
        self.pi1 = tf.distributions.Categorical(logits=self.predict_op1)
        self.pi2 = tf.distributions.Categorical(logits=self.predict_op2)
        
        self.opt = tf.train.AdamOptimizer(learning_rate=2.5e-4, beta1=0.9, beta2=0.999, epsilon=1.)
        self.train_op = self.opt.apply_gradients(grads_and_vars=self.gvs(self.states, self.actions, self.rewards, self.advantages))
        
        self.get_weights_op = self.model1.get_weights()
        self.set_weights_op = self.model2.set_weights(self.get_weights_op)
        
    def get_weights(self):
        return self.get_weights_op.run(feed_dict={})

    def set_weights(self):
        return self.set_weights_op
        
    ####################################################################

    def predict(self, state):
        value = self.predict_op1.eval(feed_dict={self.states : [state]})
        value = np.squeeze(value)
        
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, 2)
        else:
            action_idx = np.argmax(value)
            
        action = np.zeros(shape=2)
        action[action_idx] = 1
            
        return value, action
        # return value[action_idx], action_idx

    def gvs(self, states, actions, rewards, advantages):
    
        ############
    
        [pred1, forward1] = self.model1.forward(states)
        [pred2, forward2] = self.model2.forward(states)
        pred1 = pred1 + self.bias

        values1 = tf.reduce_max(pred1, axis=1)
        values2 = tf.reduce_max(pred2, axis=1)

        ############

        ratio = tf.exp(self.pi1.log_prob(actions) - self.pi2.log_prob(actions))
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = advantages * ratio
        surr2 = advantages * tf.clip_by_value(ratio, 1 - 0.1, 1 + 0.1)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi1.entropy())

        '''
        def build_network(self, name, trainable=True):
            with tf.variable_scope(name):
                conv1 = tf.layers.conv2d(self.states, 32, 8, 4, activation=tf.nn.relu, trainable=trainable)
                conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu, trainable=trainable)
                conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu, trainable=trainable)
                flattened = tf.layers.flatten(conv3)
                fc = tf.layers.dense(flattened, 512, activation=tf.nn.relu, trainable=trainable)

                values = tf.squeeze(tf.layers.dense(fc, 1, trainable=trainable), axis=-1)
                action_logits = tf.layers.dense(fc, action_space_dim, trainable=trainable)
                action_dists = tf.distributions.Categorical(logits=action_logits)

                params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

                return action_dists, values, params
        '''
        # figure out what 'values' are.

        clipped_value_estimate = values2 + tf.clip_by_value(values1 - values2, -0.1, 0.1)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, rewards)
        value_loss_2 = tf.squared_difference(values1, rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss
        grads = tf.gradients(loss, [self.bias])
        grad = grads[0] / self.nbatch
        
        grads_and_vars = self.model1.backward(states, forward1, grad)
        return grads_and_vars

    def train(self, states, actions, rewards, advantages):
        self.train_op.run(feed_dict={self.states:states, self.actions:actions, self.rewards:rewards, self.advantages:advantages})

####################################
        
def create_model(nbatch):
    l1_1 = Convolution(input_sizes=[nbatch, 80, 80, 4], filter_sizes=[8, 8, 4, 32], init='sqrt_fan_in', strides=[1,4,4,1], padding="SAME", name='conv1')
    l1_2 = BatchNorm(input_size=[nbatch, 20, 20, 32], name='conv1_bn')
    l1_3 = Relu()
    l1_4 = MaxPool(size=[nbatch, 20, 20, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = Convolution(input_sizes=[nbatch, 10, 10, 32], filter_sizes=[4, 4, 32, 64], init='sqrt_fan_in', strides=[1,2,2,1], padding="SAME", name='conv2')
    l2_2 = BatchNorm(input_size=[nbatch, 5, 5, 64], name='conv2_bn')
    l2_3 = Relu()

    l3_1 = Convolution(input_sizes=[nbatch, 5, 5, 64], filter_sizes=[3, 3, 64, 64], init='sqrt_fan_in', strides=[1,1,1,1], padding="SAME", name='conv3')
    l3_2 = BatchNorm(input_size=[nbatch, 5, 5, 64], name='conv3_bn')
    l3_3 = Relu()

    l4 = ConvToFullyConnected(input_shape=[5, 5, 64])

    l5_1 = FullyConnected(input_shape=5*5*64, size=512, init='sqrt_fan_in', name='fc1')
    l5_2 = BatchNorm(input_size=[nbatch, 512], name='fc1_bn')
    l5_3 = Relu()

    l6 = FullyConnected(input_shape=512, size=2, init='sqrt_fan_in', name='fc2')

    model = Model(layers=[l1_1, l1_2, l1_3, l1_4, \
                          l2_1, l2_2, l2_3,       \
                          l3_1, l3_2, l3_3,       \
                          l4,                     \
                          l5_1, l5_3,             \
                          # l5_1, l5_2, l5_3,       \
                          l6,                     \
                          ])
                          
    return model

####################################
        
        
        
        
