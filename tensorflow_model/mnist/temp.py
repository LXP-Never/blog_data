# Author:凌逆战
# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from mnist import mnist_data



train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = \
        mnist_data.prepare_MNIST_data(True)

print(train_size)               # 275000
print(train_total_data.shape)           # (275000, 794)

train_data_ = train_total_data[:, :-10]         # (275000, 784)
train_labels_ = train_total_data[:, -10:]       # (275000, 10)

print(train_data_.shape)
print(train_labels_.shape)


