# Author:凌逆战
# -*- encoding:utf-8 -*-

from ops import *
import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
w = tf.Variable(tf.constant(0.0))

global_steps = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 2, staircase=False)
loss = tf.pow(w * x - y, 2)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         _, lr, steps = sess.run([train_op, learning_rate, global_steps],
#                                 feed_dict={x: np.linspace(1, 2, 10).reshape([10, 1]),
#                                            y: np.linspace(1, 2, 10).reshape([10, 1])})
#         print(lr)
#         print(steps)
#         save_path = saver.save(sess, save_path="./log/model.cpkt", global_step=steps)
#         print("Save to path: ", save_path)

with tf.Session() as sess:
    # 加载模型，如果模型存在返回 是否加载成功和训练步数
    could_load, checkpoint_step = load_model(sess, saver, "./log")
    if could_load:
        # global_steps = tf.Variable(checkpoint_step, trainable=False)
        print(" [*] 加载成功")
    else:
        print(" [!] 加载失败")
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
    for i in range(10):
         _,lr,steps = sess.run([train_op, learning_rate, global_steps],
                               feed_dict={x: np.linspace(1, 2, 10).reshape([10, 1]),
                                          y: np.linspace(1, 2, 10).reshape([10, 1])})
         print(lr)
         print(steps)
         print("___________________")
