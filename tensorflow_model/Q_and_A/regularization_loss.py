# Author:凌贤鹏
# -*- coding:utf-8 -*-
import tensorflow as tf

# weights = tf.get_variable('weight', [None, 512],
#                           initializer=tf.truncated_normal_initializer(stddev=0.1))
# regularizer = tf.contrib.layers.l2_regularizer(0.0001)  # l2正则项 0.0001是正则化衰减
# regularization_loss = regularizer(weights)
# tf.add_to_collection('losses', regularization_loss)  # 收集正则化损失
# tf.add_to_collection('losses', mse_loss)  # 收集均方误差损失
# losses_total = tf.add_n(tf.get_collection('losses'))    # 得到总损失

conv_weights = tf.get_variable(name='weights',
                               shape=[kernel_size, kernel_size, input_size, output_size],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                               regularizer=tf.contrib.layers.l2_regularizer(0.00004))  # 正则损失衰减率0.000004



