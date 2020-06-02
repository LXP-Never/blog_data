# Author:凌逆战
# -*- encoding:utf-8 -*-
import tensorflow as tf
import os
import re


def _activation_summary(x):
    """Helper： 收集变量，增加在TensorBoard中观察模型参数"""
    # 如果多GPU训练，从name中删除“tower_u[0-9]/”。这有助于在tensorboard上清晰显示。
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)  # 收集高维度的变量
    # 稀疏性，tf.nn.zero_fraction返回0元素占总数量的比例
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def load_model(sess, saver, checkpoint_dir):
    """加载模型，
    如果模型存在，返回True和模型的step
    如果模型不存在，返回False并设置step=0"""

    # 通过checkpoint找到模型文件名
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 返回最新的chechpoint文件名 model.ckpt-1000
        print("新的chechpoint文件名", ckpt_name)  # model.ckpt-1000
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))  # 1000
        print(" [*] 成功恢复模型 {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] 找不到checkpoint")
        return False, 0


def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)


def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def batch_convert2int(images):
    """
    Args:
      images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def batch_convert2float(images):
    """
    Args:
      images: 4D int tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D float tensor
    """
    return tf.map_fn(convert2float, images, dtype=tf.float32)


def _add_loss_summaries(total_loss):
    # 计算所有单个损失和总损失的移动平均值。
    # 采用滑动平均的方法更新参数，衰减速率=0.9（更新的速度）
    # 对模型参数进行平均得到的模型往往比单个模型的结果要好很多
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    losses = tf.get_collection('losses')  # losses的列表
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op



def moving_average(total_loss, learning_rate, global_step, ):
    # 采用滑动平均的方法更新参数，moving averages移动平均值
    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):  # 控制计算指定，只有执行了括号中的语句才能执行下面的语句
        opt = tf.train.GradientDescentOptimizer(learning_rate)  # 创建优化器
        grads = opt.compute_gradients(total_loss)   # 计算梯度

    # 应用梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 为trainable variables添加histograms
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 为gradients添加 histograms
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track(跟踪) 所有 trainable variables 的moving averages
    # dacay=min(decay, (1 + num_updates) / (10 + num_updates))
    # 采用滑动平均的方法更新参数
    variable_averages = tf.train.ExponentialMovingAverage(decay=0.999, num_updates=global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        # tf.no_op()表示执行完apply_gradient_op, variable_averages_op操作之后什么都不做
        train_op = tf.no_op(name='train')

    return train_op
