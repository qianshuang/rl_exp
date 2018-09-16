# -*- coding: utf-8 -*-

import gym
import numpy as np

env = gym.make("CartPole-v0")


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
