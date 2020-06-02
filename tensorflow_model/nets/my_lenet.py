# Author:凌贤鹏
# -*- coding:utf-8 -*-
import tensorflow as tf


def conv(input, kernel_size, output_size, stride, init_bias=0.0, padding="SAME", name=None, wd=None):
    input_size = input.shape[-1]
    conv_weights = tf.get_variable(name='weights',
                                   shape=[kernel_size, kernel_size, input_size, output_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   dtype=tf.float32)
    conv_biases = tf.get_variable(name='biases',
                                  shape=[output_size],
                                  initializer=tf.constant_initializer(init_bias),
                                  dtype=tf.float32)

    if wd is not None:
        # wd 0.004
        # tf.nn.l2_loss(var)=sum(t**2)/2
        weight_decay = tf.multiply(tf.nn.l2_loss(conv_weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    conv_layer = tf.nn.conv2d(input, conv_weights, [1, stride, stride, 1], padding=padding, name=name)  # 卷积操作
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)  # 加上偏置项
    conv_layer = tf.nn.relu(conv_layer)  # relu激活函数

    return conv_layer


def fc(input, output_size, init_bias=0.0, activeation_func=True, wd=None):
    input_shape = input.get_shape().as_list()
    # 创建 全连接权重 变量
    fc_weights = tf.get_variable(name="weights",
                                 shape=[input_shape[-1], output_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1),
                                 dtype=tf.float32)
    if wd is not None:
        # wd 0.004
        # tf.nn.l2_loss(var)=sum(t**2)/2
        weight_decay = tf.multiply(tf.nn.l2_loss(fc_weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    # 创建 全连接偏置 变量
    fc_biases = tf.get_variable(name="biases",
                                shape=[output_size],
                                initializer=tf.constant_initializer(init_bias),
                                dtype=tf.float32)

    fc_layer = tf.matmul(input, fc_weights)  # 全连接计算
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)  # 加上偏置项
    if activeation_func:
        fc_layer = tf.nn.relu(fc_layer)  # rele激活函数
    return fc_layer


# 训练时：keep_prob=0.5
# 测试时：keep_prob=1.0
def leNet(inputs, class_num, keep_prob=0.5):
    # 第一层 卷积层 conv1
    with tf.variable_scope('layer1-conv1'):
        conv1 = conv(input=inputs, kernel_size=5, output_size=32, stride=1, init_bias=0.0, name="layer1-conv1",
                     padding="SAME")
    # 第二层 池化层
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第三层 卷积层 conv2
    with tf.variable_scope('layer3-conv2'):
        conv2 = conv(input=pool1, kernel_size=5, output_size=64, stride=1, init_bias=0.0, name="layer3-conv2",
                     padding="SAME")
    # 第四层 池化层
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 后面要做全连接，因此要把数据变成2维
    # pool_shape = pool2.get_shape().as_list()
    pool_shape = pool2.shape
    flatten = tf.reshape(pool2, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
    with tf.variable_scope('layer5-fcl'):
        fc1 = fc(input=flatten, output_size=512, init_bias=0.1, activeation_func=tf.nn.relu, wd=None)
        fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob, name="dropout1")
    with tf.variable_scope('layer6-fc2'):
        logit = fc(input=fc1, output_size=class_num, init_bias=0.1, activeation_func=False, wd=None)
    return logit
