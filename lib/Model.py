
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

class Model:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        
    def num_params(self):
        param_sum = 0
        for ii in range(self.num_layers):
            l = self.layers[ii]
            param_sum += l.num_params()
        return param_sum
        
    def get_weights(self):
        weights = {}
        for ii in range(self.num_layers):
            l = self.layers[ii]
            tup = l.get_weights()
            for (key, value) in tup:
                weights[key] = value
            
        return weights

    def set_weights(self, weight_dic):
        rets = []
        for ii in range(self.num_layers):
            l = self.layers[ii]
            ret = l.set_weights(weight_dic)
            rets.extend(ret)
            
        return rets

    ####################################################################

    def predict(self, state):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(state)
            else:
                A[ii] = l.forward(A[ii-1])
                
        return A[self.num_layers-1]

    ####################################################################

    def forward(self, state):
        return self.predict(state)

    def backward(self, state, action, reward, forward):
        A = forward
        D = [None] * self.num_layers
        grads_and_vars = []
        
        p_reward = tf.multiply(A[self.num_layers-1], action)
        a_reward = tf.reshape(reward, (-1, 1))
        a_reward = tf.multiply(a_reward, action)
        e_reward = p_reward - a_reward
        
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii] = l.backward(A[ii-1], A[ii], e_reward)
                gvs = l.gv(A[ii-1], A[ii], e_reward)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.backward(state, A[ii], D[ii+1])
                gvs = l.gv(state, A[ii], D[ii+1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.backward(A[ii-1], A[ii], D[ii+1])
                gvs = l.gv(A[ii-1], A[ii], D[ii+1])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars

        
        
        
        
        
        
