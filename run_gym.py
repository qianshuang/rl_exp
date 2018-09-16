# -*- coding: utf-8 -*-

import gym

# 创建一个CartPole的环境模拟器，可以向左或向右加速推车，以平衡放置在其顶部的杆
env = gym.make("CartPole-v0")
# 初始化环境变量，返回第一个观察结果
obs = env.reset()
# 每个值的含义分别是：小车的水平位置(0.0 = center)、速度、杆的角度(0.0 = 垂直)、杆的角速度
print(obs)  # [-0.03446547 -0.04519292  0.01626575 -0.01777462]
# 渲染并显示环境
env.render()

# 可能采取的行动只能是两个离散的值，向左(0)和向右(1)
print(env.action_space)  # Discrete(2)
# step表示执行一步动作，这里向右移动一步
action = 1
obs, reward, done, info = env.step(action)
# 采取行动后的下一个观测结果：向右为+，向左为-
print(obs)  # [ 0.01969743  0.23745748 -0.02107486 -0.26251706]
# 采取行动后的奖励，不管采取什么动作，奖励都是1，所以我们的目标是尽量让小车运行的时间长
print(reward)  # 1.0
# 当执行完所有episode后才会返回True。当小车倾斜角度太大，游戏结束也会返回True，这时环境必须reset才能重用
print(done)  # False
# debug信息
print(info)  # {}
