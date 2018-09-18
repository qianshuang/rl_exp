# -*- coding: utf-8 -*-

import tensorflow as tf


class TCNNConfig(object):
    """配置参数"""
    n_inputs = 4
    n_outputs = 1

    learning_rate = 1e-2    # 学习率

    n_iterations = 100         # number of training iterations
    n_max_steps = 1000  # max steps per episode
    n_games_per_update = 10  # train the policy every 10 episodes
    save_iterations = 10  # save the model every 10 training iterations
    discount_rate = 0.95


class TextCNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None, self.config.n_inputs], name='input_x')
        self.policy_gradients()

    def policy_gradients(self):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        hidden = tf.contrib.layers.fully_connected(self.input_x, 4, activation_fn=tf.nn.elu, weights_initializer=initializer)
        logits = tf.contrib.layers.fully_connected(hidden, self.config.n_outputs, activation_fn=None, weights_initializer=initializer)
        outputs = tf.nn.sigmoid(logits)
        # 计算向左和向右的概率
        p_left_and_right = tf.concat([outputs, 1 - outputs], axis=1)
        # 根据概率采样action
        self.action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
        # the target probability must be 1.0 if the chosen action is action 0 (left) and 0.0 if it is action 1 (right)
        y = 1. - tf.to_float(self.action)
        # 我们认为选择的action就是最好的action
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)
        # 计算梯度
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        self.gradients = [grad for grad, variable in grads_and_vars]
        self.gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:  # variable是NN policy的参数矩阵W和b（作为整体）的变量名
            # gradient_placeholder用来传入调整后的梯度值，即gradients * action_score->标准化->平均后的新的梯度值
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            # Tensor("Placeholder:0", shape=(4, 4), dtype=float32)
            # Tensor("Placeholder_1:0", shape=(4,), dtype=float32)
            # Tensor("Placeholder_2:0", shape=(4, 1), dtype=float32)
            # Tensor("Placeholder_3:0", shape=(1,), dtype=float32)
            print(gradient_placeholder)  # 同grad的shape
            self.gradient_placeholders.append(gradient_placeholder)
            # 将调整后的梯度值feed给优化器，以执行优化
            grads_and_vars_feed.append((gradient_placeholder, variable))
        self.training_op = optimizer.apply_gradients(grads_and_vars_feed)
