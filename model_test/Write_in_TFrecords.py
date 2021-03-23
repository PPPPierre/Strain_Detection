##将数据转换成tfrecords格式
import os
import tensorflow as tf
from PIL import Image
#import matplotlib.pyplot as plt
#import numpy as np

'''数据转换TFrecords'''
sess = tf.InteractiveSession()
classes = {'up', 'low'}  # 预先自己定义的类别，根据自己的需要修改

## Train data
cwd_train = "/home/stone/Y3S2Deeplearning/Data/Test/"  # 数据所在目录位置
writer = tf.python_io.TFRecordWriter("train.tfrecords")  # train表示转成的tfrecords数据格式的名字

for index, name in enumerate(classes):
    class_path = cwd_train + name + "/"
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

## Test data
cwd_test = "/home/stone/Y3S2Deeplearning/Data/Test_/"  # 数据所在目录位置
writer = tf.python_io.TFRecordWriter("test.tfrecords")  # train表示转成的tfrecords数据格式的名字

for index, name in enumerate(classes):
    class_path = cwd_test + name + "/"
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

## coord = tf.train.Coordinator()  # 线程管理
## threads = tf.train.start_queue_runners(sess=sess, coord=coord)

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()  # 线程管理
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in range(4):
        image,label=sess.run([image_batch,label_batch])
        shape = sess.run(tf.shape(image))
        print(shape)

    coord.request_stop()
    coord.join(threads)

