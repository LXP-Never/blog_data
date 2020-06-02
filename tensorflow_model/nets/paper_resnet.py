# Author:凌逆战
# -*- encoding:utf-8 -*-
import tensorflow as tf


def batch_normalization(inputs, output_size):
    mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])  # 计算均值和方差
    beta = tf.get_variable('beta', output_size, tf.float32, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', output_size, tf.float32, initializer=tf.ones_initializer)
    bn_layer = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

    return bn_layer


def conv(input, kernel_size, output_size, stride, padding="SAME", wd=None):
    input_size = input.shape[-1]
    conv_weights = tf.get_variable(name='weights',
                                   shape=[kernel_size, kernel_size, input_size, output_size],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                   regularizer=tf.contrib.layers.l2_regularizer(0.00004)) # 正则损失衰减率0.000004

    conv_layer = tf.nn.conv2d(input, conv_weights, [1, stride, stride, 1], padding=padding)  # 卷积操作
    batch_norm = batch_normalization(conv_layer, output_size)
    conv_output = tf.nn.relu(batch_norm)  # relu激活函数
    return conv_output


def block(input, n, output_size, change_first_stride, bottleneck):
    if n == 0 and change_first_stride:
        stride = 2
    else:
        stride = 1
    if bottleneck:
        with tf.variable_scope('a'):
            conv_a = conv(input=input, kernel_size=1, output_size=output_size, stride=stride, padding="SAME")
            conv_a = batch_normalization(conv_a, output_size)
            conv_a = tf.nn.relu(conv_a)
        with tf.variable_scope('b'):
            conv_b = conv(input=conv_a, kernel_size=3, output_size=output_size, stride=1, padding="SAME")
            conv_b = batch_normalization(conv_b, output_size)
            conv_b = tf.nn.relu(conv_b)

        with tf.variable_scope('c'):
            conv_c = conv(input=conv_b, kernel_size=1, output_size=output_size * 4, stride=1, padding="SAME")
            output = batch_normalization(conv_c, output_size * 4)
    else:
        with tf.variable_scope('A'):
            conv_A = conv(input=input, kernel_size=3, output_size=output_size, stride=stride, padding="SAME")
            conv_A = batch_normalization(conv_A, output_size)
            conv_A = tf.nn.relu(conv_A)

        with tf.variable_scope('B'):
            conv_B = conv(input=conv_A, kernel_size=3, output_size=output_size, stride=1, padding="SAME")
            output = batch_normalization(conv_B, output_size)

    if input.shape == output.shape:
        with tf.variable_scope('shortcut'):
            shortcut = input  # shortcut
    else:
        with tf.variable_scope('shortcut'):
            shortcut = conv(input=input, kernel_size=1, output_size=output_size * 4, stride=1, padding="SAME")
            shortcut = batch_normalization(shortcut, output_size * 4)

    return tf.nn.relu(output + shortcut)


def fc(input, output_size, activeation_func=True):
    input_shape = input.shape[-1]
    # 创建 全连接权重 变量
    fc_weights = tf.get_variable(name="weights",
                                 shape=[input_shape, output_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 dtype=tf.float32,
                                 regularizer=tf.contrib.layers.l2_regularizer(0.01))
    # 创建 全连接偏置 变量
    fc_biases = tf.get_variable(name="biases",
                                shape=[output_size],
                                initializer=tf.zeros_initializer,
                                dtype=tf.float32)

    fc_layer = tf.matmul(input, fc_weights)  # 全连接计算
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)  # 加上偏置项
    if activeation_func:
        fc_layer = tf.nn.relu(fc_layer)  # rele激活函数
    return fc_layer


def inference(inputs, class_num, num_blocks=[3, 4, 6, 3], bottleneck=True):
    # data[1, 224, 224, 3]

    # 我们尝试搭建50层ResNet
    with tf.variable_scope('conv1'):
        conv1 = conv(input=inputs, kernel_size=7, output_size=64, stride=2, padding="SAME")
        conv1 = batch_normalization(inputs=conv1, output_size=64)
        conv1 = tf.nn.relu(conv1)

    with tf.variable_scope('conv2_x'):
        conv_output = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        for n in range(num_blocks[0]):
            with tf.variable_scope('block%d' % (n + 1)):
                conv_output = block(conv_output, n, output_size=64, change_first_stride=False, bottleneck=bottleneck)

    with tf.variable_scope('conv3_x'):
        for n in range(num_blocks[1]):
            with tf.variable_scope('block%d' % (n + 1)):
                conv_output = block(conv_output, n, output_size=128, change_first_stride=True, bottleneck=bottleneck)

    with tf.variable_scope('conv4_x'):
        for n in range(num_blocks[2]):
            with tf.variable_scope('block%d' % (n + 1)):
                conv_output = block(conv_output, n, output_size=256, change_first_stride=True, bottleneck=bottleneck)

    with tf.variable_scope('conv5_x'):
        for n in range(num_blocks[3]):
            with tf.variable_scope('block%d' % (n + 1)):
                conv_output = block(conv_output, n, output_size=512, change_first_stride=True, bottleneck=bottleneck)

    output = tf.reduce_mean(conv_output, reduction_indices=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        output = fc(output, class_num, activeation_func=False)

    return output
