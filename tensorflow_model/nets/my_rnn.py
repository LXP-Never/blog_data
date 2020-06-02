import tensorflow as tf


#######################################
#       单层 静态/动态 LSTM/GRU        #
#######################################
# 单层静态LSTM
def single_layer_static_lstm(input_x, time_steps, hidden_size):
    """
    :param input_x: 输入张量 形状为[batch_size, n_steps, input_size]
    :param n_steps: 时序总数
    :param n_hidden: LSTM单元输出的节点个数 即隐藏层节点数
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x1 = tf.unstack(input_x, num=time_steps, axis=1)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)  # 创建LSTM_cell
    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    output, states = tf.nn.static_rnn(cell=lstm_cell, inputs=input_x1, dtype=tf.float32)  # 通过cell类构建RNN

    return output, states


# 单层静态gru
def single_layer_static_gru(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size, n_steps, input_size]
    :param n_steps: 时序总数
    :param n_hidden: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回静态单层GRU单元的输出，以及cell状态
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x = tf.unstack(input, num=time_steps, axis=1)
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)  # 创建GRU_cell
    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    output, states = tf.nn.static_rnn(cell=gru_cell, inputs=input_x, dtype=tf.float32)  # 通过cell类构建RNN

    return output, states


# 单层动态LSTM
def single_layer_dynamic_lstm(input, time_steps, hidden_size):
    """
    :param input_x: 输入张量 形状为[batch_size, time_steps, input_size]
    :param time_steps: 时序总数
    :param hidden_size: LSTM单元输出的节点个数 即隐藏层节点数
    :return: 返回动态单层LSTM单元的输出，以及cell状态
    """
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)  # 创建LSTM_cell
    # 动态rnn函数传入的是一个三维张量，[batch_size,time_steps, input_size]  输出也是这种形状
    output, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, dtype=tf.float32)  # 通过cell类构建RNN
    output = tf.transpose(output, [1, 0, 2])  # 注意这里输出需要转置  转换为时序优先的
    return output, states


# 单层动态gru
def single_layer_dynamic_gru(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size, time_steps, input_size]
    :param time_steps: 时序总数
    :param hidden_size: GRU单元输出的节点个数 即隐藏层节点数
    :return: 返回动态单层GRU单元的输出，以及cell状态
    """
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)  # 创建GRU_cell
    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,input_size]  输出也是这种形状
    output, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input, dtype=tf.float32)  # 通过cell类构建RNN
    output = tf.transpose(output, [1, 0, 2])  # 注意这里输出需要转置  转换为时序优先的
    return output, states


#######################################
#       多层 静态/动态 LSTM/GRU        #
#######################################
# 多层静态LSTM网络
def multi_layer_static_lstm(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,time_steps,input_size]
    :param time_steps: 时序总数
    :param n_hidden: LSTM单元输出的节点个数 即隐藏层节点数
    :return: 返回静态多层LSTM单元的输出，以及cell状态
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x1 = tf.unstack(input, num=time_steps, axis=1)

    # 多层RNN的实现 例如cells=[cell1,cell2,cell3]，则表示一共有三层
    mcell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size) for _ in range(3)])

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    output, states = tf.nn.static_rnn(cell=mcell, inputs=input_x1, dtype=tf.float32)

    return output, states


# 多层静态GRU
def multi_layer_static_gru(input, time_steps, hidden_size):
    """
    :param input_x: 输入张量 形状为[batch_size,n_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回静态多层GRU单元的输出，以及cell状态
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x = tf.unstack(input, num=time_steps, axis=1)

    # 多层RNN的实现 例如cells=[cell1,cell2,cell3]，则表示一共有三层
    mcell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.GRUCell(num_units=hidden_size) for _ in range(3)])

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    output, states = tf.nn.static_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    return output, states


# 多层静态GRU和LSTM 混合
def multi_layer_static_mix(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,n_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回静态多层GRU和LSTM混合单元的输出，以及cell状态
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x = tf.unstack(input, num=time_steps, axis=1)

    # 可以看做2个隐藏层
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size * 2)

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell, gru_cell])

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    output, states = tf.nn.static_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    return output, states


# 多层动态LSTM
def multi_layer_dynamic_lstm(input, time_steps, hidden_size):
    """
    :param input: 输入张量  形状为[batch_size,n_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: LSTM单元输出的节点个数 即隐藏层节点数
    :return: 返回动态多层LSTM单元的输出，以及cell状态
    """
    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size) for _ in range(3)])

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,input_size]  输出也是这种形状
    output, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    output = tf.transpose(output, [1, 0, 2])
    return output, states


# 多层动态GRU
def multi_layer_dynamic_gru(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,n_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回动态多层GRU单元的输出，以及cell状态
    """
    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.GRUCell(num_units=hidden_size) for _ in range(3)])

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,input_size]  输出也是这种形状
    output, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    output = tf.transpose(output, [1, 0, 2])
    return output, states


# 多层动态GRU和LSTM 混合
def multi_layer_dynamic_mix(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,n_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回动态多层GRU和LSTM混合单元的输出，以及cell状态
    """
    # 可以看做2个隐藏层
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size * 2)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell, gru_cell])

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,input_size]  输出也是这种形状
    output, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    output = tf.transpose(output, [1, 0, 2])
    return output, states


#######################################
#   单层/多层 双向 静态/动态 LSTM/GRU   #
#######################################
# 单层静态双向LSTM
def single_layer_static_bi_lstm(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,time_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: LSTM单元输出的节点个数 即隐藏层节点数
    :return: 返回单层静态双向LSTM单元的输出，以及cell状态
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x = tf.unstack(input, num=time_steps, axis=1)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)  # 正向
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)  # 反向

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    # 这里的输出output是一个list 每一个元素都是前向输出,后向输出的合并
    output, fw_state, bw_state = tf.nn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                                cell_bw=lstm_bw_cell,
                                                                inputs=input_x,
                                                                dtype=tf.float32)
    print(type(output))  # <class 'list'>
    print(len(output))  # 28
    print(output[0].shape)  # (?, 256)

    return output, fw_state, bw_state


# 单层动态双向LSTM
def single_layer_dynamic_bi_lstm(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,time_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回单层动态双向LSTM单元的输出，以及cell状态
    """
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)  # 正向
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)  # 反向

    # 动态rnn函数传入的是一个三维张量，[batch_size,time_steps,input_size]  输出是一个元组 每一个元素也是这种形状
    output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                    cell_bw=lstm_bw_cell,
                                                    inputs=input,
                                                    dtype=tf.float32)
    print(type(output))  # <class 'tuple'>
    print(len(output))  # 2
    print(output[0].shape)  # (?, 28, 128)
    print(output[1].shape)  # (?, 28, 128)

    output = tf.concat(output, axis=2)  # 按axis=2合并 (?,28,128) (?,28,128)按最后一维合并(?,28,256)
    output = tf.transpose(output, [1, 0, 2])  # 注意这里输出需要转置  转换为时序优先的

    return output, state


# 多层静态双向LSTM
def multi_layer_static_bi_lstm(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,time_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: LSTM单元输出的节点个数 即隐藏层节点数
    :return: 返回多层静态双向LSTM单元的输出，以及cell状态
    """
    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list
    # 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x = tf.unstack(input, num=time_steps, axis=1)

    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        stacked_fw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size))  # 正向
        stacked_bw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size))  # 反向

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,input_size)大小的张量
    # 这里的输出output是一个list 每一个元素都是前向输出,后向输出的合并
    output, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_rnn(stacked_fw_rnn,
                                                                        stacked_bw_rnn,
                                                                        inputs=input_x,
                                                                        dtype=tf.float32)
    print(type(output))  # <class 'list'>
    print(len(output))  # 28
    print(output[0].shape)  # (?, 256)

    return output, fw_state, bw_state


# 多层动态双向LSTM
def multi_layer_dynamic_bi_lstm(input, time_steps, hidden_size):
    """
    :param input: 输入张量 形状为[batch_size,n_steps,input_size]
    :param time_steps: 时序总数
    :param hidden_size: gru单元输出的节点个数 即隐藏层节点数
    :return: 返回多层动态双向LSTM单元的输出，以及cell状态
    """
    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        stacked_fw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size))  # 正向
        stacked_bw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size))  # 反向

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,input_size]  输出也是这种形状，
    # input_size变成了正向和反向合并之后的 即input_size*2
    output, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(stacked_fw_rnn,
                                                                                stacked_bw_rnn,
                                                                                inputs=input,
                                                                                dtype=tf.float32)
    print(type(output))  # <class 'tensorflow.python.framework.ops.Tensor'>
    print(output.shape)  # (?, 28, 256)

    output = tf.transpose(output, [1, 0, 2])  # 注意这里输出需要转置  转换为时序优先的

    return output, fw_state, bw_state


def RNN_inference(inputs, class_num, time_steps, hidden_size):
    """
    :param inputs: [batch_size, n_steps, input_size]
    :param class_num: 类别数
    :param time_steps: 时序总数
    :param n_hidden: LSTM单元输出的节点个数 即隐藏层节点数
    """
    #######################################
    #       单层 静态/动态 LSTM/GRU        #
    #######################################
    # outputs, states = single_layer_static_lstm(inputs, time_steps, hidden_size)  # 单层静态LSTM
    # outputs, states = single_layer_static_gru(inputs, time_steps, hidden_size)   # 单层静态gru
    # outputs, states = single_layer_dynamic_lstm(inputs, time_steps, hidden_size)  # 单层动态LSTM
    # outputs, states = single_layer_dynamic_gru(inputs, time_steps, hidden_size)  # 单层动态gru
    #######################################
    #       多层 静态/动态 LSTM/GRU        #
    #######################################
    # outputs, states = multi_layer_static_lstm(inputs, time_steps, hidden_size)  # 多层静态LSTM网络
    # outputs, states = multi_layer_static_gru(inputs, time_steps, hidden_size)  # 多层静态GRU
    # outputs, states = multi_layer_static_mix(inputs, time_steps, hidden_size)  # 多层静态GRU和LSTM 混合
    # outputs, states = multi_layer_dynamic_lstm(inputs, time_steps, hidden_size)  # 多层动态LSTM
    # outputs, states = multi_layer_dynamic_gru(inputs, time_steps, hidden_size)  # 多层动态GRU
    # outputs, states = multi_layer_dynamic_mix(inputs, time_steps, hidden_size)  # 多层动态GRU和LSTM 混合
    #######################################
    #   单层/多层 双向 静态/动态 LSTM/GRU  #
    #######################################
    # outputs, fw_state, bw_state = single_layer_static_bi_lstm(inputs, time_steps, hidden_size)  # 单层静态双向LSTM
    # outputs, state = single_layer_dynamic_bi_lstm(inputs, time_steps, hidden_size)  # 单层动态双向LSTM
    # outputs, fw_state, bw_state = multi_layer_static_bi_lstm(inputs, time_steps, hidden_size)  # 多层静态双向LSTM
    outputs, fw_state, bw_state = multi_layer_dynamic_bi_lstm(inputs, time_steps, hidden_size)  # 多层动态双向LSTM

    # output静态是 time_step=28个(batch=128, output=128)组成的列表
    # output动态是 (time_step=28, batch=128, output=128)
    print('hidden:', outputs[-1].shape)  # 最后一个时序的shape(128,128)

    # 取LSTM最后一个时序的输出，然后经过全连接网络得到输出值
    fc_output = fc(input=outputs[-1], output_size=class_num, activeation_func=tf.nn.relu)

    return fc_output


time_steps = 28
class_num = 10
input_size = 28
hidden_size = 128

# 定义占位符
# batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  input_size：表示一个时序具体的数据长度
# 即一共28个时序，一个时序送入28个数据进入LSTM网络
input_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, input_size])
input_y = tf.placeholder(dtype=tf.float32, shape=[None, class_num])

logits = RNN_inference(input_x, class_num, time_steps, hidden_size)
