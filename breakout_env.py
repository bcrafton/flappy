

import cv2
import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

env = gym.make('BreakoutNoFrameskip-v4')
action_dim = {}

for _ in range(100):
    action = env.action_space.sample()
    action_dim[action] = action
    print (action)
    
print (action_dim)
