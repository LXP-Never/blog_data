# coding:utf-8
import numpy as np
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
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def calc_grad_tiled(img, t_grad, tile_size=512):
    """可以对任意大小的图像计算梯度
    :param img: 初始化图片
    :param t_grad: 优化目标(score)对输入图片的梯度
    :param tile_size: 每次只对tile_size×tile_size大小的图像计算梯度，避免内存问题
    :return: 返回梯度更新后的图像
    """
    sz = tile_size  # 512
    h, w = img.shape[:2]
    # 防止在tile的边缘产生边缘效应对图片进行整体移动
    # 产生两个(0,sz]之间均匀分布的整数值
    sx, sy = np.random.randint(sz, size=2)
    # 先在水平方向滚动sx个位置，再在垂直方向上滚动sy个位置
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    # x, y是开始位置的像素
    for y in range(0, max(h - sz // 2, sz), sz):  # 垂直方向
        for x in range(0, max(w - sz // 2, sz), sz):  # 水平方向
            # 每次对sub计算梯度。sub的大小是tile_size×tile_size
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    # 使用np.roll滚动回去
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def resize_ratio(img, ratio):
    """将图片img放大ratio倍"""
    min = img.min()  # 图片的最小值
    max = img.max()  # 图片的最大值
    img = (img - min) / (max - min) * 255  # 归一化
    # 把输出缩放为0~255之间的数
    print("魔", img.shape)
    img = np.float32(scipy.misc.imresize(img, ratio))
    print("鬼", img.shape)
    img = img / 255 * (max - min) + min  # 将像素值缩放回去
    return img


def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    """生成更大尺寸的图像
    :param t_obj:卷积层某个通道的值
    :param img0:初始化噪声图像
    :param iter_n:迭代数
    :param step:学习率
    :param octave_n: 放大一共会进行octave_n-1次
    :param octave_scale: 图片放大倍数，大于1的"浮点数"则会变成原来的倍数！整数会变成百分比
    :return:
    """
    # 同样定义目标和梯度
    t_score = tf.reduce_mean(t_obj)  # 定义优化目标
    t_grad = tf.gradients(t_score, t_input)[0]  # 计算t_score对t_input的梯度

    img = img0.copy()
    print("原始尺寸",img.shape)
    for octave in range(octave_n):
        if octave > 0:
            # 将小图片放大octave_scale倍
            # 共放大octave_n - 1 次
            print("前", img.shape)
            img = resize_ratio(img, octave_scale)
            print("后", img.shape)
        for i in range(iter_n):
            # 调用calc_grad_tiled计算任意大小图像的梯度
            g = calc_grad_tiled(img, t_grad)    # 对图像计算梯度
            g /= g.std() + 1e-8
            img += g * step
    savearray(img, 'multiscale.jpg')


if __name__ == '__main__':
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    render_multiscale(layer_output[:, :, :, channel], img_noise, iter_n=20)
