# Author:凌贤鹏
# -*- coding:utf-8 -*-
import tensorflow as tf


# tf.train.GradientDescentOptimizer()
# 实例化 滑动平均类
# ema = tf.train.ExponentialMovingAverage(decay, num_updates=None, zero_debias=False, name='ExponentialMovingAverage')
# averages_op=ema.apply([v1]) # 计算影子变量
# averages_op.average(v1) # 获取影子变量


v1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(tf.constant(0))

ema = tf.train.ExponentialMovingAverage(0.99, num_updates=global_step)  # 实例化滑动平均模型 类
variables_average_op = ema.apply([v1])  # 计算v1的影子变量op（给v1滑动平均操作）

with tf.Session() as sess:
    init = tf.global_variables_initializer().run()
    # 变量、影子变量
    print(sess.run([v1, ema.average(v1)]))  # 初始值 [0.0, 0.0]

    sess.run(tf.assign(v1, 5))  # 相当于 v1=5
    sess.run(variables_average_op)  # v1变量滑动平均操作
    # 变量、影子变量
    print(sess.run([v1, ema.average(v1)]))  # [5.0, 4.5]
    # decay=min(0.99, 1/10)=0.1, v1=0.1*0+0.9*5=4.5

    sess.run(tf.assign(global_step, 10000))  # steps=10000
    sess.run(tf.assign(v1, 10))  # v1=10
    sess.run(variables_average_op)  # 计算因子变量 op
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.555]
    # decay=min(0.99,(1+10000)/(10+10000))=0.99, v1=0.99*4.5+0.01*10=4.555

    sess.run(variables_average_op)  # 计算因子变量 op
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.60945]
    # decay=min(0.99,(1+10000)/(10+10000))=0.99, v1=0.99*4.555+0.01*10=4.60945




