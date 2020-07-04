## 待做事务

- 考虑要不要在tensorboad中显示样本，那样的话可能就需要在训练的过程中，生成样本，参考21项目第8章代码，可能需要用到reuse_variable

- 以后把model（inference）写成一个类吧，大家都这么写，参考21项目第8、11章代码

- 定义训练步骤时指定var_list=variable_to_train。定义不会训练损失网络

`train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)
_, loss_t, step = sess.run([train_op, loss, global_step])`



重新创建一个github分支

能不能把relu等激活函数单独拎出来




