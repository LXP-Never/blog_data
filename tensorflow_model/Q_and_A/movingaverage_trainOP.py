# Author:凌贤鹏
# -*- coding:utf-8 -*-
import tensorflow as tf


def train(total_loss, global_step):
    lr = tf.train.exponential_decay(0.01, global_step, decay_steps=350, decay_rate=0.1, staircase=True)
    # 采用滑动平均的方法更新损失值
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    losses = tf.get_collection('losses')  # losses的列表
    loss_averages_op = loss_averages.apply(losses + [total_loss])  # 计算损失值的影子变量op

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):  # 控制计算指定，只有执行了括号中的语句才能执行下面的语句
        opt = tf.train.GradientDescentOptimizer(lr)  # 创建优化器
        grads = opt.compute_gradients(total_loss)  # 计算梯度

    # 应用梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 采用滑动平均的方法更新参数
    variable_averages = tf.train.ExponentialMovingAverage(0.999, num_updates=global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        # tf.no_op()表示执行完apply_gradient_op, variable_averages_op操作之后什么都不做
        train_op = tf.no_op(name='train')

    return train_op


