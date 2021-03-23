##将数据转换成tfrecords格式
import os
import tensorflow as tf
from PIL import Image
#import matplotlib.pyplot as plt
#import numpy as np

'''数据转换TFrecords'''
sess = tf.Session()
cwd = "/home/stone/Y3S2Deeplearning/Data/Test/"  # 数据所在目录位置
classes = {'up', 'low'}  # 预先自己定义的类别，根据自己的需要修改
writer = tf.python_io.TFRecordWriter("train.tfrecords")  # train表示转成的tfrecords数据格式的名字

for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        shape = tf.shape(img)
        print(sess.run(shape))
