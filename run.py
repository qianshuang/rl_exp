# -*- coding: utf-8 -*-

from policy_gradients_model import *
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
        all_rewards = []  # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        all_loss = []
        for game in range(config.n_games_per_update):
            current_rewards = []  # all raw rewards from the current episode
            current_gradients = []  # all gradients from the current episode
            current_loss = []

            obs = env.reset()
            for step in range(config.n_max_steps):
                action_val, gradients_val, loss_val = sess.run(
                    [model.action, model.gradients, model.loss],
                    feed_dict={model.input_x: obs.reshape(1, config.n_inputs)})  # one obs
                obs, reward, done, info = env.step(action_val[0][0])
                # env.render()  # render方法比较耗时
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                current_loss.append(loss_val)

                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
            all_loss.append(current_loss)

        # At this point we have run the policy for 10 episodes, and we are ready for a policy update using the algorithm described earlier.
        all_rewards_discount = discount_and_normalize_rewards(all_rewards, config.discount_rate)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(model.gradient_placeholders):
            # multiply the gradients by the action scores, and compute the mean
            compute_gradients = []  # W1:[n_games_per_update, 4, 4] b1:[n_games_per_update, 4] W2:[n_games_per_update, 4, 1] b2:[n_games_per_update, 1]
            for game_index, rewards in enumerate(all_rewards_discount):
                for step, reward in enumerate(rewards):
                    compute_gradient = reward * all_gradients[game_index][step][var_index]
                    compute_gradients.append(compute_gradient)
            mean_gradients = np.mean(compute_gradients, axis=0)  # 按位取平均
            # 下面是一步到位的写法，可读性较差
            # mean_gradients = np.mean(
            #     [reward * all_gradients[game_index][step][var_index]
            #      for game_index, rewards in enumerate(all_rewards)
            #      for step, reward in enumerate(rewards)],
            #     axis=0)
            # print(mean_gradients)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(model.training_op, feed_dict=feed_dict)

        if iteration % config.save_iterations == 0:
            saver.save(sess, save_path=save_path)

        # debug info
        time_dif = get_time_dif(start_time)
        avg_step = get_avg_step(all_rewards)  # 平均坚持多少步
        loss_train = get_avg_loss(all_loss)
        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Avg Steps: {2:>6},Time: {3}'
        print(msg.format(iteration + 1, loss_train, avg_step, time_dif))


if __name__ == '__main__':
    print('Configuring model...')
    # 构建模型计算图
    config = TCNNConfig()
    model = TextCNN(config)

    env = gym.make("CartPole-v0")
    train()
