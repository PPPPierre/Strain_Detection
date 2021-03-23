# from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

# CNN模型
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

        '''
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')
        '''

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)          ## 压平 不知道什么意思
        x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x
######################################################################################


## TFrecords读取部分
def read_and_decode(filename, batch_size):
    # 使用tf.train.match_filenames_once获取文件列表
    files = tf.train.match_filenames_once(filename)
    # 通过tf.train.string_input_producer创建输入队列，文件列表为files
    # shuffle为False避免随机打乱读文件的顺序，只会打乱文件列表的顺序，文件内部样例输出顺序不变
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    # 读取并解析一个样本
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string),
            })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [1280, 720, 3])  # 图像归一化大小
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5   #图像减去均值处理，根据自己的需要决定要不要加上
    label = tf.cast(features['label'], tf.int32)
    # 特殊处理，去数据的batch，如果不要对数据做batch处理，也可以把下面这部分不放在函数里
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    num_threads=64,
                                                    capacity=200,
                                                    min_after_dequeue=150)
    return img_batch, tf.reshape(label_batch, [batch_size])


## 测试评估部分
def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    for it in range(test_iteration):
        test_batch_image , test_batch_label = sess.run([image_batch, label_batch])
        test_feed_dict = {
            x: test_batch_image,
            label: test_batch_label,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / test_iteration
        test_acc += acc_ / test_iteration

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
    return test_acc, test_loss, summary

##########################################################################################

# Hyperparameter 超参数设置
class_num = 2               # 分类数量
growth_k = 12               # 初始图片特征图数量
nb_block = 2                # dense block 单元内bottleneck 个数
init_learning_rate = 1e-4   # 初始学习率
epsilon = 1e-4              # AdamOptimizer 的 epsilon 参数
dropout_rate = 0.2          # dropOut层参数

'''
# Momentum Optimizer 最优化数值
nesterov_momentum = 0.9     #
weight_decay = 1e-4         # 下降指数
'''

# Label & batch_size
batch_size = 1             # batch大小
train_iteration = 6             # 迭代次数（训练完所有数据的批数）
# batch_size * iteration = data_set_number

test_iteration = 6         # 测试轮数
# test_batch_size * test_iteration = test_set_number
total_epochs = 1          # 总训练次数


## 定义占位符
# 图像大小 = 1280*720, 图像通道数 = 3, 分类数量 = 2
x = tf.placeholder(tf.float32, shape=[None, 1280, 720, 3],name='x_holder')   # 要有名字，因为后续保存模型再使用，需要读取出来重新给占位符赋值
label = tf.placeholder(tf.int32, [batch_size])                           # 标签数据不是独热编码，如果标签数据是独热编码shape=[batch_size,ClassesNumber]
#label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)                             # 判定是否在训练
learning_rate = tf.placeholder(tf.float32, name='learning_rate')    # 学习率

logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model              # 模型设定
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))            # softmax加交叉熵损失函数
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))      # softmax加交叉熵损失函数
# 这里交叉熵的结果是一个维度为batch_Size的vector，reduce_mean函数的作用是求所有元素的均值。

"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
论文用的是 MomentumOptimizer
init_learning_rate = 0.1
这里用 AdamOptimizer
"""

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)    ## 最优化
train = optimizer.minimize(cost)                                                    ## 最小化cost

label_comp = tf.cast(label, tf.int64)
correct_prediction = tf.equal(tf.argmax(logits, 1), label_comp) # argmax取每次输出结果概率最大的那一类别的索引与label进行比较
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))       # reduce_mean 计算平均值
# 这里tf.cast函数的作用是数据类型的转换

## 训练数据与测试数据读取
file_path = './data.tfrecords-*'      #已经存下的tfrecords文件
image_batch, label_batch = read_and_decode(file_path,batch_size)


''' 开始训练 '''
## 模型保存
#saver_path = '/home/stone/Y3S2Deeplearning/ModelSave/DenseNet/Model_DenseNet'               # 保存路径
saver_path = './model/dense.ckpt'
saver = tf.train.Saver(tf.global_variables())   # 模型保存
model_path = './model'
summary_path = './logs'

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)     ## 读取之前的训练数据
    else:
        sess.run(tf.global_variables_initializer())         ## 参数初始化

    tf.local_variables_initializer().run() # 使用tf.train.match_filenames_once()函数需要初始化
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    epoch_learning_rate = init_learning_rate
    coord = tf.train.Coordinator()  # 线程管理
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(1,total_epochs+1):
        ## 设置学习率的退化
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, train_iteration + 1):

            batch_x, batch_y = sess.run([image_batch, label_batch])

            ## 喂入训练数据
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)      # 训练
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)                    # 计算一个batch里的准确率

            train_loss += batch_loss    # 计算总损失
            train_acc += batch_acc      # 计算总准确率

        train_loss /= train_iteration # 计算这轮训练的average loss 平均损失
        train_acc /= train_iteration  # 计算这轮训练的average accuracy 平均准确率

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_acc, test_loss, test_summary = Evaluate(sess) #计算该轮训练完后模型在测试集上的表现

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print(line)

        with open('logs.txt', 'a') as f:
            f.write(line)

        print ("training finish!")
        saver.save(sess=sess, save_path=saver_path) # 保存训练完的参数

    coord.request_stop()
    coord.join(threads)

"""
# 测试使用
test_image, test_label = sess.run([image_batch, label_batch])  # 读取数据
true_count = 0
prediction = sess.run([top_k_op],
                      feed_dict={x: test_image, label: test_label, training_flag: false})  # 计算预测和真值有多少个相同的
print(logits)
print(prediction)
true_count += np.sum(prediction)
print(true_count / total_number)  # total_number是batch_size的大小，因为每次测试都是投进去一个batch的
"""