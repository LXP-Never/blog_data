# Author:凌逆战
# -*- encoding:utf-8 -*-
import tensorflow as tf


def batch_normalization_layer(inputs, output_size):
    mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])  # 计算均值和方差
    beta = tf.get_variable('beta', output_size, tf.float32, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', output_size, tf.float32, initializer=tf.ones_initializer)
    bn_layer = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

    return bn_layer


def bn_relu_conv_layer(input, kernel_size, output_size, stride, padding="SAME"):
    """ bn --> relu --> conv """
    input_size = input.shape[-1]

    bn_layer = batch_normalization_layer(input, input_size)
    relu_layer = tf.nn.relu(bn_layer)
    filter = tf.get_variable(name='conv',
                             shape=[kernel_size, kernel_size, input_size, output_size],
                             dtype=tf.float16,
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
    output = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding=padding)
    return output


def conv_bn_relu_layer(input, kernel_size, output_size, stride, padding="SAME"):
    """conv --> bn --> relu"""
    input_size = input.shape[-1]
    conv_weights = tf.get_variable(name='conv',
                                   shape=[kernel_size, kernel_size, input_size, output_size],
                                   dtype=tf.float16,
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
    conv_layer = tf.nn.conv2d(input, conv_weights, strides=[1, stride, stride, 1], padding=padding)
    bn_layer = batch_normalization_layer(conv_layer, output_size)
    output = tf.nn.relu(bn_layer)
    return output


def conv(input, kernel_size, output_size, stride, padding="SAME"):
    input_size = input.shape[-1]
    conv_weights = tf.get_variable(name='weights',
                                   shape=[kernel_size, kernel_size, input_size, output_size],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                   regularizer=tf.contrib.layers.l2_regularizer(0.00004))
    conv_layer = tf.nn.conv2d(input, conv_weights, [1, stride, stride, 1], padding=padding)  # 卷积操作
    batch_norm = batch_normalization_layer(conv_layer, output_size)
    conv_output = tf.nn.relu(batch_norm)  # relu激活函数
    return conv_output


def residual_block(input, output_size, down_sample,  projection=False):
    input_size = input.shape[-1]

    # 当需要“缩小”image size时，我们使用stride = 2
    if down_sample:
        stride = 2
        input = tf.nn.max_pool(input, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

    conv1 = conv(input=input, kernel_size=3, output_size=output_size, stride=1)
    conv2 = conv(input=conv1, kernel_size=3, output_size=output_size, stride=1)

    if input_size != output_size:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv(input=input, kernel_size=1, output_size=output_size, stride=2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_size - input_size]])
    else:
        input_layer = input

    res = conv2 + input_layer
    return res


def resnet(inputs, depth, class_num):
    if depth < 20 or (depth - 20) % 12 != 0:
        print("ResNet 的深度不对！")
        return
    conv_num = (depth - 20) / 12 + 1
    layers = []

    with tf.variable_scope("conv1"):
        conv1 = conv(input=inputs, kernel_size=3, output_size=16, stride=1)
        layers.append(conv1)

    for i in range(conv_num):
        with tf.variable_scope("conv2_%d" % (i + 1)):
            conv2_x = residual_block(input=layers[-1], output_size=16, down_sample=False)
            conv2 = residual_block(input=conv2_x, output_size=16, down_sample=False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.shape[1:] == [32, 32, 16]

    for i in range(conv_num):
        down_sample = True if i == 0 else False
        with tf.variable_scope("conv3_%d" % (i + 1)):
            conv3_x = residual_block(input=layers[-1], output_size=32, down_sample=down_sample)
            conv3 = residual_block(input=conv3_x, output_size=32, down_sample=False)
            layers.append(conv3_x)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

    for i in range(conv_num):
        down_sample = True if i == 0 else False
        with tf.variable_scope("conv4_%d" % (i + 1)):
            conv4_x = residual_block(input=layers[-1], output_size=64, down_sample=down_sample)
            conv4 = residual_block(input=conv4_x, output_size=64, down_sample=False)
            layers.append(conv4_x)
            layers.append(conv4)
        assert conv4.shape[1:] == [8, 8, 64]

    with tf.variable_scope('fc'):
        input_size = layers[-1].shape[-1]
        bn_layer = batch_normalization_layer(inputs=layers[-1], output_size=input_size)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.shape[1:] == [64]

        fc_weights = tf.get_variable(name="fc_weights",
                                     shape=[global_pool.shape[-1], class_num],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
        fc_biases = tf.get_variable(name='fc_bias',
                                    shape=[class_num],
                                    initializer=tf.zeros_initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
        fc_h = tf.matmul(global_pool, fc_weights) + fc_biases
        layers.append(fc_h)

    return layers[-1]







