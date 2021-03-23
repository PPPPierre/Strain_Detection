import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm, flatten
#from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope

## 卷积层
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input,
                                   use_bias=False,      ##无偏置值
                                   filters=filter,
                                   kernel_size=kernel,
                                   strides=stride,
                                   padding='SAME')      ##补零
        return network


##  全局平均池化层 （global average pooling）
def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x,
                                       pool_size=pool_size,
                                       strides=stride)


## BN层
def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


## dropout层
def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x,
                             rate=rate,
                             training=training)


## 激活函数：线性整流
def Relu(x):
    return tf.nn.relu(x)


## 平均池化层
def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x,
                                       pool_size=pool_size,
                                       strides=stride,
                                       padding=padding)


## 最大池化层
def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


## 特征图合并
def Concatenation(layers):
    return tf.concat(layers, axis=3)


## 全连接层： 功能相当于从tensor转到全连接层 units表示全连接层神经元个数
def Linear(x):
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

## 构建整个DenseNet
class DenseNet():
    ## 初始化
    def __init__(self, x, nb_blocks, filters, training):
        ##参数输入
        self.nb_blocks = nb_blocks      ## 单元层（denseblock + transition）个数
        self.filters = filters          ## 卷积核个数
        self.training = training        ## 是否在训练
        self.model = self.Dense_net(x)  ## DenseNet网络模型

    ## 定义瓶颈层
    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            # 1*1的卷积 将特征图数量翻四倍
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # 3*3 的卷积 特征图数量不变
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    ## 定义过渡层
    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

            # https://github.com/taki0112/Densenet-Tensorflow/issues/10

            # 1*1的卷积将特征图数量减少一半， 减少参数
            in_channel = x.shape[-1]
            x = conv_layer(x, filter=in_channel * 0.5, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    ## 定义denseBlock
    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()                  # 创建一个list用来存所有的特征图
            layers_concat.append(input_x)           # 将最早的输入x放进去

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)                 # 将结果叠加进list

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)    # 把所有的特征图叠加作为输入给后一层
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    ## 定义DenseNet
    def Dense_net(self, input_x):
        ## 先进行一个 7*7 的卷积
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """

        ## 交替densevlock和transition层
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)          ## 压平 不知道什么意思
        x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x