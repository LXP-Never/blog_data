# Author:凌逆战
# -*- encoding:utf-8 -*-
import tensorflow as tf


def conv(inputs, scope_name, kernel_size, output_size, stride, init_bias=0.0, padding="SAME", wd=None):
    input_size = int(inputs.get_shape()[-1])
    with tf.variable_scope(scope_name):
        conv_weights = tf.get_variable(name='weights',
                                       shape=[kernel_size, kernel_size, input_size, output_size],
                                       dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1e-1))
        if wd is not None:
            # tf.nn.l2_loss(var)=sum(t**2)/2
            weight_decay = tf.multiply(tf.nn.l2_loss(conv_weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        conv_biases = tf.get_variable(name='biases',
                                      shape=[output_size],
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(init_bias))
        conv_layer = tf.nn.conv2d(inputs, conv_weights, [1, stride, stride, 1], padding=padding, name=scope_name)
        conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
        conv_layer = tf.nn.relu(conv_layer)
    return conv_layer


def fc(inputs, scope_name, output_size, init_bias=0.0, activeation_func=True, wd=None):
    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(scope_name):
        # 创建 全连接权重 变量
        fc_weights = tf.get_variable(name="weights",
                                     shape=[input_shape[-1], output_size],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1e-1))
        if wd is not None:
            # wd 0.004
            # tf.nn.l2_loss(var)=sum(t**2)/2
            weight_decay = tf.multiply(tf.nn.l2_loss(fc_weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        # 创建 全连接偏置 变量
        fc_biases = tf.get_variable(name="biases",
                                    shape=[output_size],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(init_bias),
                                    trainable=True)

        fc_layer = tf.matmul(inputs, fc_weights)  # 全连接计算
        fc_layer = tf.nn.bias_add(fc_layer, fc_biases)  # 加上偏置项
        if activeation_func:
            fc_layer = tf.nn.relu(fc_layer)  # rele激活函数
    return fc_layer


def VGG16Net(inputs, class_num):
    with tf.variable_scope("conv1"):
        # conv1_1 [conv3_64]
        conv1_1 = conv(inputs=inputs, scope_name="conv1_1", kernel_size=3, output_size=64, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv1_2 [conv3_64]
        conv1_2 = conv(inputs=conv1_1, scope_name="conv1_2", kernel_size=3, output_size=64, stride=1,
                       init_bias=0.0, padding="SAME")
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    with tf.variable_scope("conv2"):
        # conv2_1
        conv2_1 = conv(inputs=pool1, scope_name="conv2_1", kernel_size=3, output_size=128, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv2_2
        conv2_2 = conv(inputs=conv2_1, scope_name="conv2_2", kernel_size=3, output_size=128, stride=1,
                       init_bias=0.0, padding="SAME")
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    with tf.variable_scope("conv3"):
        # conv3_1
        conv3_1 = conv(inputs=pool2, scope_name="conv3_1", kernel_size=3, output_size=256, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv3_2
        conv3_2 = conv(inputs=conv3_1, scope_name="conv3_2", kernel_size=3, output_size=256, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv3_3
        conv3_3 = conv(inputs=conv3_2, scope_name="conv3_3", kernel_size=3, output_size=256, stride=1,
                       init_bias=0.0, padding="SAME")
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    with tf.variable_scope("conv4"):
        # conv4_1
        conv4_1 = conv(inputs=pool3, scope_name="conv4_1", kernel_size=3, output_size=512, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv4_2
        conv4_2 = conv(inputs=conv4_1, scope_name="conv4_2", kernel_size=3, output_size=512, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv4_3
        conv4_3 = conv(inputs=conv4_2, scope_name="conv4_3", kernel_size=3, output_size=512, stride=1,
                       init_bias=0.0, padding="SAME")
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    with tf.variable_scope("conv5"):
        # conv5_1
        conv5_1 = conv(inputs=pool4, scope_name="conv4_1", kernel_size=3, output_size=512, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv5_2
        conv5_2 = conv(inputs=conv5_1, scope_name="conv4_2", kernel_size=3, output_size=512, stride=1,
                       init_bias=0.0, padding="SAME")
        # conv5_3
        conv5_3 = conv(inputs=conv5_2, scope_name="conv4_3", kernel_size=3, output_size=512, stride=1,
                       init_bias=0.0, padding="SAME")
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    input_shape = pool5.get_shape().as_list()  # 后面做全连接，所以要把shape改成2维
    # shape=[batch, dim]
    flatten = tf.reshape(pool5, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
    fc1 = fc(inputs=flatten, scope_name="fc1", output_size=4096, init_bias=1.0, activeation_func=True)
    fc2 = fc(inputs=fc1, scope_name="fc2", output_size=4096, init_bias=1.0, activeation_func=True)
    fc3 = fc(inputs=fc2, scope_name="fc3", output_size=class_num, init_bias=1.0, activeation_func=True)

    return fc3
