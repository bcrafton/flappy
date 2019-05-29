
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

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
        self.y = tf.placeholder("float", [None]) # does this = value ... i think its reward.
        self.advantages = tf.placeholder("float", [None])
        
        self.model1 = create_model()
        self.model2 = create_model()
        
    def get_weights(self):
        return self.model1.get_weights()

    def set_weights(self):
        return self.model2.set_weights(self.model1.get_weights())
        
    ####################################################################

    def predict(self, state):
        return self.model1.predict(state)

    def gvs(self, state, action, reward, advantage):
    
        ############
    
        forward1 = self.model1.forward(?) + self.bias
        forward2 = self.model2.forward(?)

        ############

        ratio = tf.exp(self.pi.log_prob(self.actions) - self.old_pi.log_prob(self.actions))
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - 0.1, 1 + 0.1)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi.entropy())

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.values - self.old_values, -0.1, 0.1)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.values, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        loss = policy_loss + entropy_coef * entropy_loss + vf_coef * value_loss
        grads = tf.gradients(loss, [self.bias])
        grad = grads[0] / self.nbatch
        
        ############
        
        grads_and_vars = self.model1.backward(forward1, grad)

        ############

        return grads_and_vars

####################################
        
def create_model():
    l1_1 = Convolution(input_sizes=[args.batch_size, 80, 80, 4], filter_sizes=[8, 8, 4, 32], init='sqrt_fan_in', strides=[1,4,4,1], padding="SAME", name='conv1', load=weights_conv)
    l1_2 = BatchNorm(input_size=[args.batch_size, 20, 20, 32], name='conv1_bn', load=weights_conv)
    l1_3 = Relu()
    l1_4 = MaxPool(size=[args.batch_size, 20, 20, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = Convolution(input_sizes=[args.batch_size, 10, 10, 32], filter_sizes=[4, 4, 32, 64], init='sqrt_fan_in', strides=[1,2,2,1], padding="SAME", name='conv2', load=weights_conv)
    l2_2 = BatchNorm(input_size=[args.batch_size, 5, 5, 64], name='conv2_bn', load=weights_conv)
    l2_3 = Relu()

    l3_1 = Convolution(input_sizes=[args.batch_size, 5, 5, 64], filter_sizes=[3, 3, 64, 64], init='sqrt_fan_in', strides=[1,1,1,1], padding="SAME", name='conv3', load=weights_conv)
    l3_2 = BatchNorm(input_size=[args.batch_size, 5, 5, 64], name='conv3_bn', load=weights_conv)
    l3_3 = Relu()

    l4 = ConvToFullyConnected(input_shape=[5, 5, 64])

    l5_1 = FullyConnected(input_shape=5*5*64, size=512, init='sqrt_fan_in', name='fc1', load=weights_fc)
    l5_2 = BatchNorm(input_size=[args.batch_size, 512], name='fc1_bn', load=weights_fc)
    l5_3 = Relu()

    l6 = FullyConnected(input_shape=512, size=2, init='sqrt_fan_in', name='fc2', load=weights_fc)

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
        
        
        
        
