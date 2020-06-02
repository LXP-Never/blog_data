# Author:凌逆战
# -*- encoding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
from nets.slim_mnist_model import CNN
from ops import *

tf.flags.DEFINE_integer('batch_size', 50, 'batch size, default: 1')
tf.flags.DEFINE_integer('class_num', 10, 'batch size, default: 1')
tf.flags.DEFINE_integer('epochs', 10, 'batch size, default: 1')
tf.flags.DEFINE_string('checkpoints_dir', "checkpoints", '保存检查点的地址')
FLAGS = tf.flags.FLAGS

# 从MNIST_data/中读取MNIST数据。当数据不存在时，会自动执行下载
mnist = input_data.read_data_sets('./data', one_hot=True, reshape=False)

# 将数组张换成图片形式
print(mnist.train.images.shape)  # 训练数据图片(55000, 28, 28, 1)
print(mnist.train.labels.shape)  # 训练数据标签(55000, 10)
print(mnist.test.images.shape)  # 测试数据图片(10000, 28, 28, 1)
print(mnist.test.labels.shape)  # 测试数据图片(10000, 10)
print(mnist.validation.images.shape)  # 验证数据图片(5000, 28, 28, 1)
print(mnist.validation.labels.shape)  # 验证数据图片(5000, 10)


def eval():
    batch_size = FLAGS.batch_size
    batch_nums = mnist.train.images.shape[0] // batch_size  # 一个epoch中应该包含多少batch数据
    class_num = FLAGS.class_num
    test_batch_size = 5000
    test_batch_num = mnist.test.images.shape[0] // test_batch_size

    ############    保存检查点的地址   ############
    checkpoints_dir = FLAGS.checkpoints_dir  # checkpoints
    # 如果检查点不存在，则创建
    if not os.path.exists(checkpoints_dir):
        print("模型文件不存在，无法进行评估")

    ######################################################
    #                    创建图                          #
    ######################################################
    graph = tf.Graph()  # 自定义图
    # 在自己的图中定义数据和操作
    with graph.as_default():
        is_training = tf.placeholder(tf.bool, name='MODE')  # Boolean for MODE of train or test
        inputs = tf.placeholder(dtype="float", shape=[None, 28, 28, 1], name='inputs')
        labels = tf.placeholder(dtype="float", shape=[None, class_num], name='labels')
        ############    搭建模型   ############
        logits = CNN(inputs, class_num, is_training=is_training)  # 使用placeholder搭建模型
        ############    模型精度   ############
        predict = tf.argmax(logits, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))
        ############    模型保存和恢复 Saver   ############
        saver = tf.train.Saver(max_to_keep=5)

    ######################################################
    #                   创建会话                          #
    ######################################################
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=config, graph=graph) as sess:
        # 加载模型，如果模型存在返回 是否加载成功和训练步数
        could_load, checkpoint_step = load_model(sess, saver, FLAGS.checkpoints_dir)
        if could_load:
            print(" [*] 模型加载成功")
        else:
            print(" [!] 模型加载失败")
            raise ValueError("模型文件不存在，无法进行评估")

        for i in range(test_batch_num):
            test_batch_x, test_batch_y = mnist.test.next_batch(test_batch_num)
            acc = sess.run(accuracy, feed_dict={inputs: test_batch_x,
                                                labels: test_batch_y,
                                                is_training: False})
            print("模型精度为：", acc)
        xxx = mnist.test.images[1].reshape(1, 28, 28, 1)
        yyy = mnist.test.labels[1].reshape(1, 10)
        print(xxx.shape, yyy.shape)  # (1, 28, 28, 1) (1, 10)
        pre_yyy = sess.run(predict, feed_dict={inputs: xxx,
                                               is_training: False})
        # print("123", tf.argmax(pre_yyy, 1).eval())  # [7]
        # print("123", tf.argmax(yyy, 1).eval())  # 7


def main(argv=None):
    eval()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
