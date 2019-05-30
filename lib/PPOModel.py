
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
        self.bias = tf.Variable(np.zeros(shape=(self.nbatch, self.nclass)), dtype=tf.float32)
        self.epsilon = epsilon
        self.decay_max = decay_max

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

        self.sample_action_op = tf.squeeze(self.pi1.sample(1), axis=0, name='sample_action')
        self.eval_action = self.pi1.mode()
        
        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)

    def get_weights(self):
        return self.get_weights_op.run(feed_dict={})

    def set_weights(self):
        self.sess.run(self.set_weights_op, feed_dict={})
        self.sess.run(self.global_step_op, feed_dict={})
        
    ####################################################################

    def predict(self, state, stochastic=True):
        if stochastic:
            action, value = self.sess.run([self.sample_action_op, self.predict_op1], {self.states: [state]})
        else:
            action, value = self.sess.run([self.eval_action, self.predict_op1], {self.states: [state]})

        value = value[0]
        action_idx = action[0]
        
        action = np.zeros(shape=2)
        action[action_idx] = 1
        
        return value, action

    def gvs(self, states, actions, rewards, advantages):
    
        ############
    
        [pred1, forward1] = self.model1.forward(states)
        [pred2, forward2] = self.model2.forward(states)
        pred1 = pred1 + self.bias

        values1 = tf.reduce_max(pred1, axis=1)
        values2 = tf.reduce_max(pred2, axis=1)

        ############
        # DOES FORWARD = SELF.PI ? 
        # OR CATEGORICAL(FORWARD) = SELF.PI ? 
        
        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)

        ratio = tf.exp(self.pi1.log_prob(actions) - self.pi2.log_prob(actions))
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = advantages * ratio
        surr2 = advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi1.entropy())

        clipped_value_estimate = values2 + tf.clip_by_value(values1 - values2, -epsilon_decay, epsilon_decay)
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

    l6 = FullyConnected(input_shape=512, size=2, name='fc2')

    model = Model(layers=[l1_1, l1_2, l1_3, \
                          l2_1, l2_2,       \
                          l3_1, l3_2,       \
                          l4,               \
                          l5_1, l5_2,       \
                          l6,               \
                          ])
                          
    return model

####################################
        
        
        
        
