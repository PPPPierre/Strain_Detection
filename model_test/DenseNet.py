import os
from PIL import Image
import tensorflow as tf
import numpy as np
#from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
#from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
import DensenetModle as ds

'''数据转换TFrecords'''
sess = tf.InteractiveSession()
cwd = "/home/stone/Y3S2Deeplearning/Data/Test/"  # 数据所在目录位置
classes = {'up', 'low'}  # 预先自己定义的类别，根据自己的需要修改
writer = tf.python_io.TFRecordWriter("train.tfrecords")  # train表示转成的tfrecords数据格式的名字

for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((300, 300))  # 图像reshape大小设置，根据自己的需要修改
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
writer.close()

'''预处理部分'''
'''
for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    print(image, label)
'''

'''TFrecords读取部分'''
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
    img = tf.reshape(img, [300, 300, 3])  # 图像归一化大小
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5   #图像减去均值处理，根据自己的需要决定要不要加上
    label = tf.cast(features['label'], tf.int32)

    # 特殊处理，去数据的batch，如果不要对数据做batch处理，也可以把下面这部分不放在函数里
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    num_threads=64,
                                                    capacity=200,
                                                    min_after_dequeue=150)
    return img_batch, tf.reshape(label_batch, [batch_size])

init = tf.initialize_all_variables()
tfrecords_file = 'train.tfrecords'   #要读取的tfrecords文件
BATCH_SIZE = 4      #batch_size的大小
image_batch, label_batch = read_and_decode(tfrecords_file,BATCH_SIZE)
print(image_batch,label_batch)    #注意，这里不是tensor，tensor需要做see.run()处理

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 线程管理
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)
    image,label=sess.run([image_batch,label_batch])
    print(image)

'''训练评估'''
def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_image = test_x
        test_batch_label = test_y
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_image,
            label: test_batch_label,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / 10.0
        test_acc += acc_ / 10.0

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

'''Hyperparameter 超参数设置'''
class_num = 10              #分类数量
growth_k = 24               # 初始图片特征图数量
nb_block = 2                # dense block 单元内bottleneck 个数
init_learning_rate = 1e-4
epsilon = 1e-4              # AdamOptimizer 的 epsilon 参数
dropout_rate = 0.2          # dropOut层参数
# Momentum Optimizer 动量最优化数值
nesterov_momentum = 0.9     #
weight_decay = 1e-4         #重力下降指数
# Label & batch_size
batch_size = 64             #batch大小
iteration = 782             # 迭代次数（训练完所有数据的批数）
# batch_size * iteration = data_set_number
test_iteration = 10         # 训练轮数
total_epochs = 300          # 总训练次数

'''定义占位符和操作函数'''
# 图像大小 = 300, 图像通道数 = 3, 分类数量 = 10
x = tf.placeholder(tf.float32, shape=[None, 300, 300, 3],name='x_holder') #要有名字，因为后续保存模型再使用，需要读取出来重新给占位符赋值
label = tf.placeholder(tf.float32, shape=[None, class_num]) #标签数据不是独热编码，如果标签数据是独热编码shape=[batch_size,ClassesNumber]

training_flag = tf.placeholder(tf.bool)     # 判定是否在训练

learning_rate = tf.placeholder(tf.float32, name='learning_rate')    #学习率

logits = ds.DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model      # 模型设定
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))        # softmax加交叉熵损失函数

"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
论文用的是 MomentumOptimizer
init_learning_rate = 0.1
这里用 AdamOptimizer
"""

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)    ## 优化
train = optimizer.minimize(cost)    ## 训练

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)) # argmax取每次输出结果概率最大的那一类别的索引与label进行比较
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))       # reduce_mean 计算平均值

saver = tf.train.Saver(tf.global_variables())   ## 模型保存
saver_path = './model/dense.ckpt' #保存路径


'''训练数据与测试数据读取'''
tfrecords_file = 'train.tfrecords'      #已经存下的tfrecords文件
tfrecords_file_test = 'test.tfrecords'
image_batch, label_batch = read_and_decode(tfrecords_file,batch_size)
test_batch_x,test_batch_y = read_and_decode(tfrecords_file_test,batch_size)

#image_train, label_train, test_x, test_y = prepare_data()
#train_x, test_x = color_preprocessing(train_x, test_x)

'''创建会话'''
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)     ## 读取之前的训练数据
    else:
        sess.run(tf.global_variables_initializer())         ## 参数初始化

    coord = tf.train.Coordinator()  # 线程管理
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        ## 设置学习率的退化
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, iteration + 1):

            '''
            ## 设定batch范围
            if pre_index + batch_size < 50000:
                batch_x = train_x[pre_index: pre_index + batch_size]
                batch_y = train_y[pre_index: pre_index + batch_size]
            else:
                batch_x = train_x[pre_index:]
                batch_y = train_y[pre_index:]
            '''



            #batch_x = data_augmentation(batch_x) # 数据增强

            batch_x, batch_y = sess.run([image_batch, label_batch])

            ## 喂入训练数据
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)      # 训练
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)                    # 计算一个batch的准确率

            train_loss += batch_loss    # 计算总损失
            train_acc += batch_acc      # 计算总准确率
            pre_index += batch_size     # 定位batch在总训练数据的位置

            if step == iteration:
                train_loss /= iteration  # average loss 平均损失
                train_acc /= iteration  # average accuracy 平均准确率

                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

                test_x, test_y = sess.run([test_batch_x, test_batch_y])
                test_acc, test_loss, test_summary = Evaluate(sess)

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
