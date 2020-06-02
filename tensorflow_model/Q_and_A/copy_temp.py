# -*- coding: utf-8 -*-
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf


# 这里添加了参数train，用于区分训练过程和测试过程。
# 程序中将用到dropout方法，可进一步提升模型的可靠性并防止过拟合，dropout过程只在训练时使用
def inference(input_tensor, train, regularizer):
    # 声明第一层卷积层的变量并实现前向传播过程。通过使用不同命名空间来隔离不同层的变量，
    # 让每一层中的变量命名只需要考虑在当前层的作用，
    # 不需担心重命名的问题。第一层输出为28×28×32的张量
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。该最大池化层卷积核边长为2，使用0填充，移动步幅为2.
    # 该层的输入为28×28×32的张量，输出为14×14×32的张量
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第三层卷积层的变量并实现前向传播过程，该卷积层的输入为14×14×32的张量，输出为14×14×64的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程，输入为14×14×64，输出为7×7×64的张量
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层为7×7×64的张量，第五层输入为向量，所以需要将该张量拉成一个向量
    # pool2.get_shape函数取第四层输出张量的维度，每层的输入输出都为一个BATCH的张量，所以这里得到的维度也包含一个BATCH中数据的数量。
    pool_shape = pool2.get_shape().as_list()

    # 计算将张量拉直成向量后的长度，该长度等于张量维度累乘。注意这里的pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape函数将第四层的输出变成一个batch的向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层全连接层的变量并实现前向传播过程。输入长度为3136的向量，输出长度为512的向量。该层引入了dropout的概念，
    # dropout在训练时随机将部分结点的输出改为0.dropout一般只在全连接层而不是卷积层或池化层使用。
    with tf.variable_scope('layer5-fcl'):
        fc1_weights = tf.get_variable('weight', [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层变量并实现前向传播，输入长度为512的向量，输出长度为10的向量。输出通过softmax之后可得到最后的分类结果。
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit


# 配置神经网络的参数
BATCH_SIZE = 8


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-output')

    regularizer = tf.contrib.layers.l2_regularizer(0.0001)  # l2正则项
    y_hat = inference(x, True, regularizer)  # 调用推断过程
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)  # 实例化移动平均类
    variables_average_op = variable_averages.apply(tf.trainable_variables())  # 计算移动平均得到影子变量op
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=tf.argmax(y, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 总损失+
    # 定义学习率
    learning_rate = tf.train.exponential_decay(0.8, global_step, mnist.train.num_examples / BATCH_SIZE, 0.99)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name='train')  # 执行完上下文之后什么都不做

    # 初始化TF持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程会有独立的过程完成
        for i in range(10000):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, 28, 28, 1))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y: ys})

            # 每1000次迭代保存一次模型
            if i % 1000 == 0:
                # 输出模型在当前训练批量下的损失函数大小
                print('After %d training steps, loss on training batch is %g.' % (step, loss_value))

                # 保存当前模型，并使用global_step 参数特定地命名
                saver.save(sess, os.path.join("./model/fcn_mnist", "fcn_mnist.ckpt"), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('./data/MNIST/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
