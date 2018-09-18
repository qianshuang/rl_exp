# -*- coding: utf-8 -*-

from loss_policy_gradients_model import *
from data.cnews_loader import *

import os
import numpy as np
import gym


save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径


def train():
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Training and evaluating...')
    start_time = time.time()

    for iteration in range(config.n_iterations):
        all_rewards = []  # all raw rewards from the current episode
        all_actions = []
        all_inputs = []

        obs = env.reset()

        while True:
            inputs = obs.reshape(1, config.n_inputs)
            action_prob = sess.run(
                [model.outputs],
                feed_dict={model.input_x: inputs})  # one obs
            a = [i for i in range(len(np.array(action_prob).ravel()))]
            p = np.array(action_prob).ravel()
            action = np.random.choice(a, p=p)
            obs_, reward, done, info = env.step(action)
            # env.render()  # render方法比较耗时
            all_rewards.append(reward)
            all_actions.append(action)
            all_inputs.append(inputs)

            if done:
                # 每一个epoch都进行训练
                discounted_ep_rs_norm = discount_and_normalize_rewards([all_rewards], config.discount_rate)[0]
                # train on episode
                loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                    model.input_x: np.vstack(all_inputs),
                    model.input_actions: all_actions,  # shape=[None, ]
                    model.input_action_scores: discounted_ep_rs_norm,  # shape=[None, ]
                })

                break

            obs = obs_

        if iteration % config.save_iterations == 0:
            saver.save(sess, save_path=save_path)

        # debug info
        time_dif = get_time_dif(start_time)
        avg_step = np.sum(all_rewards)  # 平均坚持多少步
        loss_train = loss
        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Avg Steps: {2:>6},Time: {3}'
        print(msg.format(iteration + 1, loss_train, avg_step, time_dif))


if __name__ == '__main__':
    print('Configuring model...')
    # 构建模型计算图
    config = TCNNConfig()
    model = TextCNN(config)

    env = gym.make("CartPole-v0")
    train()
