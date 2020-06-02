# Author:凌逆战
# -*- encoding:utf-8 -*-
import time
from tensorflow.examples.tutorials.mnist import input_data
from nets.my_silm_CNN import CNN
from ops import *

tf.flags.DEFINE_integer('batch_size', 50, 'batch size, default: 1')
tf.flags.DEFINE_integer('class_num', 10, 'batch size, default: 1')
tf.flags.DEFINE_integer('epochs', 10, 'batch size, default: 1')
tf.flags.DEFINE_string('checkpoints_dir', "checkpoints", '保存检查点的地址')
FLAGS = tf.flags.FLAGS

# 从MNIST_data/中读取MNIST数据。当数据不存在时，会自动执行下载
mnist = input_data.read_data_sets('./data', one_hot=True, reshape=False)
# reshape=False  (None, 28,28,1)    # 用于第一层是卷积层
# reshape=False  (None, 784)        # 用于第一层是全连接层

# 我们看一下数据的shape
print(mnist.train.images.shape)  # 训练数据图片(55000, 28, 28, 1)
print(mnist.train.labels.shape)  # 训练数据标签(55000, 10)
print(mnist.test.images.shape)  # 测试数据图片(10000, 28, 28, 1)
print(mnist.test.labels.shape)  # 测试数据图片(10000, 10)
print(mnist.validation.images.shape)  # 验证数据图片(5000, 28, 28, 1)
print(mnist.validation.labels.shape)  # 验证数据图片(5000, 784)


def train():
    batch_size = FLAGS.batch_size
    batch_nums = mnist.train.images.shape[0] // batch_size  # 一个epoch中应该包含多少batch数据
    class_num = FLAGS.class_num
    epochs = FLAGS.epochs

    ############    保存检查点的地址   ############
    checkpoints_dir = FLAGS.checkpoints_dir  # checkpoints
    # 如果检查点不存在，则创建
    if not os.path.exists(checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)

    ######################################################
    #                    创建图                          #
    ######################################################
    graph = tf.Graph()  # 自定义图
    # 在自己的图中定义数据和操作
    with graph.as_default():
        inputs = tf.placeholder(dtype="float", shape=[None, 28, 28, 1], name='inputs')
        labels = tf.placeholder(dtype="float", shape=[None, class_num], name='labels')
        keep_prob = tf.placeholder(dtype="float", name='keep_prob')  # 训练的时候需要，测试的时候需要设置成1
        ############    搭建模型   ############
        logits = CNN(inputs, class_num, keep_prob=keep_prob)  # 使用placeholder搭建模型
        ############    损失函数   ############
        # loss = slim.losses.softmax_cross_entropy(y, y_)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection("loss"))    # total_loss=模型损失+权重正则化损失
        ############    模型精度   ############
        predict = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))
        ############    优化器   ############
        variable_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  # 可训练变量列表
        # 创建优化器，更新网络参数，最小化loss，
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=1e-4,
                                                   global_step=global_step,
                                                   decay_steps=batch_nums,
                                                   decay_rate=0.1, staircase=True)
        # 移动平均值更新参数
        # train_op = moving_average(loss, learning_rate, global_step)
        # adam优化器,adam算法好像会自动衰减学习率，
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=total_loss, global_step=global_step,
                                                                  var_list=variable_to_train)
        ############    TensorBoard可视化 summary  ############
        summary_writer = tf.summary.FileWriter(logdir="./logs", graph=graph)  # 创建事件文件
        tf.summary.scalar(name="losses", tensor=total_loss)  # 收集损失值变量
        tf.summary.scalar(name="acc", tensor=accuracy)  # 收集精度值变量
        tf.summary.scalar(name='learning_rate', tensor=learning_rate)
        # tf.summary.image('X/generated', batch_convert2int(FLAGS.batch_image)) # 删除警告
        merged_summary_op = tf.summary.merge_all()  # 将所有的summary合并为一个op
        ############    模型保存和恢复 Saver   ############
        saver = tf.train.Saver(max_to_keep=5)

    ######################################################
    #                   创建会话                          #
    ######################################################
    max_acc = 0.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=config, graph=graph) as sess:
        # 加载模型，如果模型存在返回 是否加载成功和训练步数
        could_load, checkpoint_step = load_model(sess, saver, FLAGS.checkpoints_dir)
        if could_load:
            print(" [*] 模型加载成功")
        else:
            print(" [!] 模型加载失败")
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

        for epoch in range(epochs):
            for i in range(batch_nums):
                start_time = time.time()
                # batch_images = data_X[i * batch_size:(i + 1) * batch_size]
                # batch_labels = data_y[i * batch_size:(i + 1) * batch_size]
                train_batch_x, train_batch_y = mnist.train.next_batch(batch_size)

                # 使用真实数据填充placeholder，运行训练模型和合并变量操作
                _, summary, loss, step = sess.run([train_op, merged_summary_op, total_loss, global_step],
                                                  feed_dict={inputs: train_batch_x,
                                                             labels: train_batch_y,
                                                             keep_prob: 0.5})
                if step % 100 == 0:
                    summary_writer.add_summary(summary, step)  # 将每次迭代后的变量写入事件文件
                    summary_writer.flush()  # 要不要查明一下什么意思

                ############    可视化打印   ############
                print("Epoch：[%2d] [%4d/%4d] time：%4.4f，loss：%.8f" % (
                    epoch, i, batch_nums, time.time() - start_time, loss))

                # 打印一些可视化的数据，损失...
                if step % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={inputs: mnist.validation.images,
                                                        labels: mnist.validation.labels,
                                                        keep_prob: 1.0})
                    print("Epoch：[%2d] [%4d/%4d] accuracy：%.8f" % (epoch, i, batch_nums, acc))
                    ############    保存模型   ############
                    if acc > max_acc:
                        max_acc = acc
                        save_path = saver.save(sess,
                                               save_path=os.path.join(checkpoints_dir, "model.ckpt"),
                                               global_step=step)
                        tf.logging.info("模型保存在: %s" % save_path)
        print("优化完成!")


def main(argv=None):
    train()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
