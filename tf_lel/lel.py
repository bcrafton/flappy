from __future__ import print_function

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
import tensorflow as tf

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_test = x_test.reshape(TEST_EXAMPLES, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

l1 = tf.layers.dense(x, 100, activation=tf.nn.tanh)
l2 = tf.layers.dense(l1, 10)

predict = tf.argmax(l2, axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l2, labels=y))

correct = tf.equal(predict, tf.argmax(y, 1))
correct = tf.reduce_sum(tf.cast(correct, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=args.alpha, beta1=0.9, beta2=0.999, epsilon=1).minimize(loss)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(args.epochs):
    for jj in range(0, TRAIN_EXAMPLES, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([optimizer], feed_dict ={x: xs, y: ys})
        
    total_correct = 0.0

    for jj in range(0, TEST_EXAMPLES, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = x_test[s:e]
        ys = y_test[s:e]
        _correct = sess.run(correct, feed_dict ={x: xs, y: ys})
        total_correct += _correct
            
    print ("acc: %f" % (total_correct / TEST_EXAMPLES))
        
        
