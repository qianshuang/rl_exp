# -*- coding: utf-8 -*-

import numpy as np

nan = np.nan  # represents impossible actions
# shape=[s, a, s'], 即[3, 3, 3]，比如第2行第1列的[0.0, 1.0, 0.0]代表s1a0s0,s1a0s1,s1a0s2
T = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
    [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
])
R = np.array([  # shape=[s, a, s']
    [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
    [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
])
# 每个状态可能采取的action
possible_actions = [[0, 1, 2], [0, 2], [1]]
# -inf for impossible actions
Q = np.full((3, 3), -np.inf)
# Initial value = 0.0, for all possible actions
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0
# learning_rate = 0.01
discount_rate = 0.95
n_iterations = 100
for iteration in range(n_iterations):
    Q_prev = Q.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q[s, a] = np.sum([
                T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prev[sp]))
                for sp in range(3)
            ])

print(Q)  # [[21.88646117 20.79149867 16.854807], [1.10804034 -inf 1.16703135], [-inf 53.8607061 -inf]]
print(np.argmax(Q, axis=1))  # [0 2 1]，optimal action for each state
