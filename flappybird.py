
import tensorflow as tf
import numpy as np
import gym
import gym_ple
import os
import cv2
import time
from time import sleep
import multiprocessing as mp

##################################

game_name = 'FlappyBird-v0'
# env = gym.make(game_name)

action_set = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

action_space_dim = len(action_set)
learner_port = 22226
num_workers = 8
display_interval = 100
log_dir = '/tmp/flappy_bird/dppo/'

##################################

ep_max = 10000
decay_max = 8000
entropy_coef = 0.01
vf_coef = 1.0
learning_rate = 0.00025
batch_size = 128
minibatch_size = 512
epochs = 5
epsilon = 0.1

##################################

class FlappyBirdEnv:
    def __init__(self):
        self.env = gym.make(game_name)
        self.env.seed(np.random.randint(0, 100000))
        self.total_reward = 0.0
        self.total_step = 0

    def reset(self):
        state = self.env.reset()
        self.total_reward = 0.0
        self.total_step = 0
        return self._process(state)

    def step(self, action):
        cumulated_reward = 0.0
        for a in action_set[action]:
            next_state, reward, done, _ = self.env.step(a)
            cumulated_reward += self._reward_shaping(reward)
            self.total_step += 1
            if done:
                break
            self.total_reward += reward
        return self._process(next_state), cumulated_reward, done

    def _reward_shaping(self, reward):
        if  reward > 0.0:
            return 1.0
        elif reward < 0.0:
            return -1.0
        else:
            return 0.01

    def _process(self, state):
        output = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        output = output[:410, :]
        output = cv2.resize(output, (84, 84))
        output = output / 255.0
        output = np.stack([output] * 4, axis=2)
        return output

class PPO(object):
    def __init__(self, sess):
        self.sess = sess
        self.states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.advantages = tf.placeholder(tf.float32, [None], name='advantages')
        self.actions = tf.placeholder(tf.int32, [None], name='actions')

        old_pi, old_values, old_params = self.build_network('old_network', trainable=False)
        self.pi, self.values, params = self.build_network('network')
        self.sample_action_op = tf.squeeze(self.pi.sample(1), axis=0, name='sample_action')
        self.eval_action = self.pi.mode()

        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('loss'):
            epsilon_decay = tf.train.polynomial_decay(epsilon, global_step, decay_max, 0.001)
            with tf.variable_scope('policy'):
                ratio = tf.exp(self.pi.log_prob(self.actions) - old_pi.log_prob(self.actions))
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = self.advantages * ratio
                surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            with tf.variable_scope('entropy'):
                entropy_loss = -tf.reduce_mean(self.pi.entropy())

            with tf.variable_scope('critic'):
                clipped_value_estimate = old_values + tf.clip_by_value(self.values - old_values, -epsilon_decay, epsilon_decay)
                value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
                value_loss_2 = tf.squared_difference(self.values, self.rewards)
                value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

            loss = policy_loss + entropy_coef * entropy_loss + vf_coef * value_loss

        with tf.variable_scope('train'):
            learning_rate_decay = tf.train.polynomial_decay(learning_rate, global_step, decay_max, 0.000001)
            optimizer = tf.train.AdamOptimizer(learning_rate_decay)
            self.train_op = optimizer.minimize(loss, var_list=params)
            self.update_old_op = [old_p.assign(p) for p, old_p in zip(params, old_params)]
            self.global_step_op = global_step.assign_add(1)

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

    def update(self, states, rewards, advantages, actions):
        sess.run([self.update_old_op, self.global_step_op])
        inds = np.arange(batch_size*num_workers)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, len(inds), minibatch_size):
                end = start + minibatch_size
                fd = {
                    self.states: [states[i] for i in inds[start:end]],
                    self.rewards: [rewards[i] for i in inds[start:end]],
                    self.advantages: [advantages[i] for i in inds[start:end]],
                    self.actions: [actions[i] for i in inds[start:end]]
                }
                sess.run(self.train_op, fd)

    # only time we call this [evaluate_state] is for inference, not training. 
    # so mode just returns the largest thing ? which would be predict ? 
    def evaluate_state(self, state, stochastic=True):
        if stochastic:
            action, value = self.sess.run(
                [self.sample_action_op, self.values], {self.states: [state]})
        else:
            action, value = self.sess.run(
                [self.eval_action, self.values], {self.states: [state]})
        return action[0], value[0]

def returns_advantages (replay_buffer, next_value, gamma=0.99, lam=0.95):
    rewards = [rb['r'] for rb in replay_buffer]
    values = [rb['v'] for rb in replay_buffer] + [next_value]
    dones = [rb['done'] for rb in replay_buffer]

    gae = 0
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    for t in reversed(range(len(replay_buffer))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return returns, advantages

##################################

cluster = tf.train.ClusterSpec({
    'learner': ['localhost:{}'.format(learner_port)],
    'worker' : ['localhost:{}'.format(learner_port + i + 1) for i in range(num_workers)]
})

def worker(task_idx, coordinator_queue, train_data_queue):
    env = FlappyBirdEnv()

    with tf.Session(server.target) as sess:
        ppo = PPO(sess)

        for e in range(ep_max+1):
            if e == 0:
                state = env.reset()
                total_rewards = [0.0, 0.0]
                total_steps = [0, 0]

            replay_buffer = []
            for _ in range(batch_size):
                a, v = ppo.evaluate_state(state, stochastic=True)

                next_state, r, done = env.step(a)
                next_state = np.concatenate((state[:, :, 1:], next_state[:, :, -1:]), axis=2)
                if done and env.total_step >= 10000:
                    # reason why we do this is bc we need a next value if u look in return_adv
                    _, next_value = ppo.evaluate_state(next_state, stochastic=True)
                    r += 0.99 * next_value

                replay_buffer.append({'s':state, 'v':v, 'a':a, 'r':r, 'done':done})
                state = next_state
                if done:
                    total_rewards.append(env.total_reward)
                    total_steps.append(env.total_step)
                    state = env.reset()
                    
            # reason why we do this is bc we need a next value if u look in return_adv
            _, next_value = ppo.evaluate_state(next_state, stochastic=True)
            returns, advs = returns_advantages(replay_buffer, next_value)

            train_data_queue.put((
                [rb['s'] for rb in replay_buffer],
                returns,
                advs,
                [rb['a'] for rb in replay_buffer]
            ))
            
##################################

coordinator_queue = mp.Queue()
train_data_queue = mp.Queue()

workers = []
for i in range(num_workers):
    workers.append(mp.Process(target=worker, args=(i, coordinator_queue, train_data_queue), daemon=True))
    workers[-1].start()

server = tf.train.Server(cluster, job_name='learner', task_index=0)
sess = tf.Session(server.target)
ppo = PPO(sess)
sess.run(tf.global_variables_initializer())


for e in range(ep_max+1):
    start = time.time()

    for _ in range(num_workers):
        coordinator_queue.put(0)

    states, rewards, advantages, actions, summaries = [], [], [], [], []
    for _ in range(num_workers):
        batch = train_data_queue.get()
        states.extend(batch[0])
        rewards.extend(batch[1])
        advantages.extend(batch[2])
        actions.extend(batch[3])
        summaries.append(batch[4])

    ppo.update(states, rewards, advantages, actions)

    if e % display_interval == 0:
        print('Episode: {}, Elapsed Time: {:.2f}, at {}'.format(
            e,
            time.time() - start,
            time.strftime('%H:%M:%S', time.localtime()),
        ))
        for s in sorted(summaries):
            print(s)

##################################

total_rewards = []
total_steps = []
env = FlappyBirdEnv()

for e in range(200):
    state = env.reset()

    while True:
        a, _ = ppo.evaluate_state(state, stochastic=False)
        next_state, r, done = env.step(a)
        state = np.concatenate((state[:, :, 1:], next_state[:, :, -1:]), axis=2)
        if done:
            total_rewards.append(env.total_reward)
            total_steps.append(env.total_step)
            break

    print('Iter :', e, '| Score:', total_rewards[-1], '| Mean Score', round(np.mean(total_rewards), 2))

np.mean(total_rewards), np.min(total_rewards), np.max(total_rewards)



