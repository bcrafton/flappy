

import io
from collections import deque
from pathlib import Path
from typing import Dict, List, Union

import cv2
import multiprocessing
import multiprocessing.connection
import time
import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

#####

# class Orthogonal(object):
# ...

#####

# class Game(object):
# ...

#####

# class Worker(object):
# ...

#####

class Model(object):
    def __init__(self, *, reuse: bool, batch_size: int):

    self.obs = tf.placeholder(shape=(batch_size, 84, 84, 4), name="obs", dtype=np.uint8)
    obs_float = tf.to_float(self.obs, name="obs_float")
    
    with tf.variable_scope("model", reuse=reuse):
        self.h = Model._cnn(obs_float)
        self.pi_logits = Model._create_policy_network(self.h, 4)
        self.value = Model._create_value_network(self.h)
        self.params = tf.trainable_variables()
        self.action = Model._sample(self.pi_logits)
        self.neg_log_pi = self.neg_log_prob(self.action, "neg_log_pi_old")
        self.policy_entropy = Model._get_policy_entropy(self.pi_logits)

    @staticmethod
    def _get_policy_entropy(logits: tf.Tensor):
        a = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a = tf.exp(a)
        z = tf.reduce_sum(exp_a, axis=-1, keepdims=True)
        p = exp_a / z
        return tf.reduce_sum(p * (tf.log(z) - a), axis=-1)

    def neg_log_prob(self, action: tf.Tensor, name: str) -> tf.Tensor:
        one_hot_actions = tf.one_hot(action, 4)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pi_logits, labels=one_hot_actions, dim=-1, name=name)

    @staticmethod
    def _sample(logits: tf.Tensor):
        uniform = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(uniform)), axis=-1, name="action")

#####

class Trainer(object):
    def __init__(self, model: Model):
        self.model = model
        self.sampled_obs = self.model.obs
        self.sampled_action = tf.placeholder(dtype=tf.int32, shape=[None], name="sampled_action")
        self.sampled_return = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_return")
        self.sampled_normalized_advantage = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_normalized_advantage")
        self.sampled_neg_log_pi = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_neg_log_pi")
        self.sampled_value = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_value")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
        self.clip_range = tf.placeholder(dtype=tf.float32, shape=[], name="clip_range")
        
        neg_log_pi = self.model.neg_log_prob(self.sampled_action, "neg_log_pi")
        ratio = tf.exp(self.sampled_neg_log_pi - neg_log_pi, name="ratio")
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range, name="clipped_ratio")
        self.policy_reward = tf.reduce_mean(tf.minimum(ratio * self.sampled_normalized_advantage, clipped_ratio * self.sampled_normalized_advantage), name="policy_reward")

        value = self.model.value
        clipped_value = tf.add(self.sampled_value, tf.clip_by_value(value - self.sampled_value, -self.clip_range, self.clip_range), name="clipped_value")
        self.vf_loss = tf.multiply(0.5, tf.reduce_mean(tf.maximum(tf.square(value - self.sampled_return), tf.square(clipped_value - self.sampled_return))), name="vf_loss")
        self.loss = -(self.policy_reward - 0.5 * self.vf_loss + 0.01 * self.entropy_bonus)

        params = self.model.params
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, params), 0.5)

        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
        grads_and_vars = list(zip(grads, params))
        self.train_op = adam.apply_gradients(grads_and_vars, name="apply_gradients")

        self.approx_kl_divergence = .5 * tf.reduce_mean(tf.square(neg_log_pi - self.sampled_neg_log_pi))
        self.clip_fraction = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))
        
        self.train_info_labels = ['policy_reward',
                                  'value_loss',
                                  'entropy_bonus',
                                  'approx_kl_divergence',
                                  'clip_fraction']



    def train(self, session: tf.Session, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):

        feed_dict = {self.sampled_obs: samples['obs'],
                     self.sampled_action: samples['actions'],
                     self.sampled_return: samples['values'] + samples['advantages'],
                     self.sampled_normalized_advantage: Trainer._normalize(samples['advantages']),
                     self.sampled_value: samples['values'],
                     self.sampled_neg_log_pi: samples['neg_log_pis'],
                     self.learning_rate: learning_rate,
                     self.clip_range: clip_range}

        evals = [self.policy_reward,
                 self.vf_loss,
                 self.entropy_bonus,
                 self.approx_kl_divergence,
                 self.clip_fraction,
                 self.train_op]

        return session.run(evals, feed_dict=feed_dict)[:-1]

    @staticmethod
    def _normalize(adv: np.ndarray):
        return (adv - adv.mean()) / (adv.std() + 1e-8)








