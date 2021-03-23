import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

## CNN模型
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
            # 1*1的卷积 将特征图数量减少
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
            # in_channel = x.shape[-1]
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
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
        x = Max_Pooling(x, pool_size=[3,3], stride=2)

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


## TFrecords读取部分
def read_and_decode(filename, batch_size):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [300, 300, 3])                  # 图像归一化大小
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5   #图像减去均值处理
    label = tf.cast(features['label'], tf.int32)

    # 特殊处理，去数据的batch，如果不要对数据做batch处理，也可以把下面这部分不放在函数里
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    num_threads=16,
                                                    capacity=20,
                                                    min_after_dequeue=15)
    return img_batch, tf.reshape(label_batch, [batch_size])


'''超参数设置'''
class_num = 2               # 分类数量
growth_k = 24               # 初始特征图数量
nb_block = 2                # dense block 单元内bottleneck 个数
dropout_rate = 0.2          # dropOut层比率
batch_size = 1


'''定义占位符'''
training_flag = tf.placeholder(tf.bool)
x_holder = tf.placeholder(tf.float32, [batch_size, 300, 300, 3], name='x_holder')  # 要有名字，因为后续保存模型再使用，需要读取出来重新给占位符赋值
y_holder = tf.placeholder(tf.int32, [batch_size])  # 标签数据不是独热编码，如果标签数据是独热编码shape=[batch_size,ClassesNumber]

logits = DenseNet(x=x_holder, nb_blocks=nb_block, filters=growth_k, training=training_flag).model          # 将数据放进去网络中
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_holder))      # 损失函数

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)                             # 优化器
top_k_op = tf.nn.in_top_k(logits, y_holder, 1)                                                     # 在测试过程中用于计算有多少个正确的

# 读取文件
tfrecords_file = 'train.tfrecords'  # 已经存下的tfrecords文件，就在同目录下，否则会找不到
BATCH_SIZE = 1                      # batchsize大小
image_batch, label_batch = read_and_decode(tfrecords_file, BATCH_SIZE)


'''开始训练'''
# 用于保存模型
saver_path = '/home/stone/Y3S2Deeplearning/ModelSave/DenseNet/Model_DenseNet'  # 保存路径
saver = tf.train.Saver()

sess = tf.Session()
##with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()  # 线程管理
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(8):  # 迭代次数
    image, label = sess.run([image_batch, label_batch])  # 将batch转换成tensor
    print(label)
    train_feed_dict = {
        x_holder: image,
        y_holder: label,
        training_flag: True
    }
    sess.run(optimizer, feed_dict=train_feed_dict)  # 训练模型
print("training finish!")
saver.save(sess, saver_path)  # 将训练好的模型保存
