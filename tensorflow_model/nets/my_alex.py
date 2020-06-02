# Author:凌逆战
# -*- encoding:utf-8 -*-
import tensorflow as tf


def conv(input, kernel_size, output_size, stride, init_bias=0.0, padding="SAME", name=None, wd=None):
    input_size = input.shape[-1]
    conv_weights = tf.get_variable(name='weights',
                                   shape=[kernel_size, kernel_size, input_size, output_size],
                                   initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                   dtype=tf.float32)
    if wd is not None:
        # wd 0.004
        # tf.nn.l2_loss(var)=sum(t**2)/2
        weight_decay = tf.multiply(tf.nn.l2_loss(conv_weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    conv_biases = tf.get_variable(name='biases',
                                  shape=[output_size],
                                  initializer=tf.constant_initializer(init_bias),
                                  dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input, conv_weights, [1, stride, stride, 1], padding=padding, name=name)  # 卷积操作
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)  # 加上偏置项
    conv_layer = tf.nn.relu(conv_layer)  # relu激活函数

    return conv_layer


def fc(input, output_size, init_bias=0.0, activeation_func=True, wd=None):
    input_shape = input.get_shape().as_list()
    # 创建 全连接权重 变量
    fc_weights = tf.get_variable(name="weights",
                                 shape=[input_shape[-1], output_size],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
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


def LRN(input, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    """Local Response Normalization 局部响应归一化"""
    return tf.nn.local_response_normalization(input, depth_radius=depth_radius, alpha=alpha,
                                              beta=beta, bias=bias)


def alexNet(inputs, class_num, keep_prob=0.5):
    # 第一层卷积层 conv1
    with tf.variable_scope("conv1"):
        conv1 = conv(input=inputs, kernel_size=7, output_size=96, stride=3, init_bias=0.0, name="conv1", padding="SAME")
        conv1 = LRN(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")
    # 第二层卷积层 conv2
    with tf.variable_scope("conv2"):
        conv2 = conv(input=conv1, kernel_size=7, output_size=96, stride=3, init_bias=1.0, name="conv2", padding="SAME")
        conv2 = LRN(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")
    # 第三层卷积层 conv3
    with tf.variable_scope("conv3"):
        conv3 = conv(input=conv2, kernel_size=7, output_size=96, stride=3, init_bias=0.0, name="conv3", padding="SAME")
    # 第四层卷积层 conv4
    with tf.variable_scope("conv4"):
        conv4 = conv(input=conv3, kernel_size=7, output_size=96, stride=3, init_bias=1.0, name="conv4", padding="SAME")
    # 第五层卷积层 conv5
    with tf.variable_scope("conv5"):
        conv5 = conv(input=conv4, kernel_size=3, output_size=256, stride=1, init_bias=1.0, name="conv5")
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool5")
    conv5_shape = conv5.shape  # 后面做全连接，所以要把shape改成2维
    # shape=[batch, dim]
    flatten = tf.reshape(conv5, [-1, conv5_shape[1] * conv5_shape[2] * conv5_shape[3]])
    # 第一层全连接层 fc1
    with tf.variable_scope("fc1"):
        fc1 = fc(input=flatten, output_size=4096, init_bias=1.0, activeation_func=tf.nn.relu, wd=None)
        fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob, name="dropout1")
    # 第一层全连接层 fc2
    with tf.variable_scope("fc2"):
        fc2 = fc(input=fc1, output_size=4096, init_bias=1.0, activeation_func=tf.nn.relu, wd=None)
        fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob, name="dropout1")
    # 第一层全连接层 fc3
    with tf.variable_scope("fc3"):
        logit = fc(input=fc2, output_size=class_num, init_bias=1.0, activeation_func=False, wd=None)

    return logit  # 模型输出
