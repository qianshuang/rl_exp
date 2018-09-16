# -*- coding: utf-8 -*-

"""This file is just for testing"""

import numpy as np

from data.cnews_loader import *

all_rewards = [[1.0, 1.0], [1.0, 1.0, 1.0]]
# avg_step = np.mean(np.sum(all_rewards, axis=1), axis=0)  # 平均坚持多少步
# print(avg_step)  # 报错，不能变长
# print(np.mean(all_rewards))  # 报错，不能变长
print(get_avg_step(all_rewards))
print(get_avg_loss(all_rewards))
