# coding: utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def savearray(img_array, img_name):
    """把numpy.ndarray保存图片"""
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def render_naive(t_obj, img0, iter_n=20, step=1.0):
    """通过调整输入图像t_input，来让优化目标t_score尽可能的大
    :param t_obj: 卷积层某个通道的值
    :param img0:初始化噪声图像
    :param iter_n:迭代数
    :param step:学习率
    """
    # t_score是优化目标。它是t_obj的平均值
    # t_score越大，就说明神经网络卷积层对应通道的平均激活越大
    t_score = tf.reduce_mean(t_obj)
    # 计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]

    # 创建新图
    img = img0.copy()
    for i in range(iter_n):
        # 在sess中计算梯度，以及当前的score
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # 对img应用梯度。step可以看做“学习率”
        g /= g.std() + 1e-8
        img += g * step
        print('score(mean)=%f' % score)
    # 保存图片
    savearray(img, 'naive.jpg')


# 定义卷积层、通道数，并取出对应的tensor
name = 'mixed4d_3x3_bottleneck_pre_relu'
layer_output = graph.get_tensor_by_name("import/%s:0" % name)  # 该层输出为(? , ?, ? , 144)
# 因此channel可以取0~143中的任何一个整数值
channel = 139
# 定义原始的图像噪声 作为初始的图像优化起点
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
# 调用render_naive函数渲染
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=20)
