# -*- coding: utf-8 -*-

import numpy as np

import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_avg_step(all_rewards):
    return np.mean([np.sum(i) for i in all_rewards])


def get_avg_loss(all_loss):
    cnt = np.sum([len(i) for i in all_loss])
    return np.sum([np.sum(i) for i in all_loss]) / cnt


def discount_rewards(rewards, discount_rate):
    """
    :param rewards: [10, 0, -50]
    :param discount_rate: 0.8
    :return: [-22., -40., -50.]
    """
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    """
    :param all_rewards: [[10, 0, -50], [10, 20]]
    :param discount_rate: 0.8
    :return: [[-0.28435071, -0.86597718, -1.18910299], [1.26665318, 1.0727777]]
    """
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
