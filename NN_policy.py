# -*- coding: utf-8 -*-

import gym
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

env = gym.make("CartPole-v0")


# 使用神经网络预估每个action的概率
def NN_policy(obs):
    n_inputs = 4  # 使用所有的环境变量值
    n_hidden = 4  # it's a simple task, we don't need more hidden neurons
    n_outputs = 1  # only outputs the probability of accelerating left
    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
    logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)
    # 3. Select a random action based on the estimated probabilities
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
    return action


# 杆向左倾斜时向左走，向右倾斜时向右走
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


totals = [1,2,3]
for episode in range(50):
    episode_rewards = 0
    obs = env.reset()
    for step in range(100):  # 100 steps max, we don't want to run forever
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))  # 41.132075471698116 12.619298990560923 1.0 64.0
