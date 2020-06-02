# Author:凌逆战
# -*- encoding:utf-8 -*-
import tensorflow as tf


def conv(input, kernel_size, output_size, stride, init_bias=0.0, padding="SAME", name=None,
         activation_fn=tf.nn.relu, wd=None):
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
    if activation_fn is not None:
        conv_layer = activation_fn(conv_layer)  # rele激活函数
    return conv_layer


def fc(input, scope_name, output_size, init_bias=0.0, activation_fn=True, wd=None):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(scope_name):
        # 创建 全连接权重 变量
        fc_weights = tf.get_variable(name="weights",
                                     shape=[input_shape[-1], output_size],
                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
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
        if activation_fn:
            fc_layer = tf.nn.relu(fc_layer)  # rele激活函数
    return fc_layer


def CNN(input, class_num, keep_prob=0.5):
    input = tf.reshape(input, [-1, 28, 28, 1])
    with tf.variable_scope("conv1"):
        conv1 = conv(input=input, kernel_size=5, output_size=32, stride=1, init_bias=0.0, padding="SAME", name="conv1")
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")
    with tf.variable_scope("conv2"):
        conv2 = conv(input=pool1, kernel_size=5, output_size=64, stride=1, init_bias=0.0, padding="SAME", name="conv2")
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")
    pool2_shape = pool2.shape  # 后面做全连接，所以要把shape改成2维
    # shape=[batch, dim]
    flatten = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
    with tf.variable_scope("fc1"):
        fc1 = fc(input=flatten, scope_name="fc1", output_size=1024, init_bias=1.0, activation_fn=tf.nn.relu, wd=None)
        dropout1 = tf.nn.dropout(fc1, keep_prob=keep_prob, name="dropout1")
    with tf.variable_scope("fc2"):
        fc2 = fc(input=dropout1, scope_name="fc2", output_size=class_num, init_bias=1.0, activation_fn=False, wd=None)

    return fc2
