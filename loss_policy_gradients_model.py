# -*- coding: utf-8 -*-

import tensorflow as tf


class TCNNConfig(object):
    """配置参数"""
    n_inputs = 4
    n_outputs = 2

    learning_rate = 1e-2    # 学习率

    n_iterations = 500         # number of training iterations
    save_iterations = 10  # save the model every 10 training iterations
    discount_rate = 0.95


class TextCNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None, self.config.n_inputs], name='input_x')
        self.input_actions = tf.placeholder(tf.int32, [None, ], name='input_actions')
        self.input_action_scores = tf.placeholder(tf.float32, [None, ], name="input_action_scores")
        self.policy_gradients()

    def policy_gradients(self):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        hidden = tf.contrib.layers.fully_connected(self.input_x, 4, activation_fn=tf.nn.elu, weights_initializer=initializer)
        logits = tf.contrib.layers.fully_connected(hidden, self.config.n_outputs, activation_fn=None, weights_initializer=initializer)
        self.outputs = tf.nn.softmax(logits)

        # 我们认为选择的action就是最好的action
        neg_log_prob = tf.reduce_sum(-tf.log(self.outputs) * tf.one_hot(self.input_actions, self.config.n_outputs), axis=1)
        # 或者是用下面的方式，两种方式等价
        # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_actions)
        # reward guided loss，这里通过action_score的正负向来指导neg_log_prob，与通过action_score的正负向来指导梯度更新是一样的效果
        self.loss = tf.reduce_mean(neg_log_prob * self.input_action_scores)
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

