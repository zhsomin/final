# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/06/06 17:09:22
@Author  :   ZH_Somin 
@Version :   1.0
@Contact :   achouloves@163.com
'''

# here put the import lib



import cv2
import os
import numpy as np
import tensorflow as tf
import random
def thresholding_inv(image):
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 6))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    bin = cv2.medianBlur(bin, 3)
    # bin = cv2.erode(bin,kernel_erode)
    bin=cv2.dilate(bin,kernel_erode,iterations = 1)
    return bin

im = cv2.imread('./image/test_2.jpg')# 样图位置，
im_th = thresholding_inv(im)
cv2.imshow('1',im_th)
cv2.waitKey(0)

t=40
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng1= int(rect[3])
    leng2= int(rect[2])
    pt1 = int(rect[1] )
    pt2 = int(rect[0] )
    roi = im_th[pt1:pt1+leng1, pt2:pt2+leng2]
    # 生成统一尺寸图片，类似于mnist数据集
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    cv2.imshow('roi',roi)
    cv2.waitKey(1000)
    # 保存图片至相应路径
    cv2.imwrite('./datasets/test2/%s.jpg'%t,roi)
    t=t+1
# 导入数据图片，以features命名
imgs=os.listdir('./datasets/train/imgs/')
# 定义一个排序函数
def nu_str(string):
    return int(string.split('.')[0])
# 将文件夹中的文件按照名称数字大小进行排序 能够与labels一一对应
imgs.sort(key=nu_str)
features_train=[]
# 对每一张图片进行处理，主要是将矩阵转化为一个向量，最后将所有图片打包
for i in imgs:
    img=cv2.imread('./datasets/train/imgs/'+str(i),0)
    #res,img=cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    #img=cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
    #cv2.imshow('3',img)
    #cv2.waitKey(100)
    img=img.reshape(28*28)/255
    features_train.append(img)
features_train=np.array(features_train) # 包含所有图片的一个向量集

## labels
## 将每一个图片对应的结果转化为one-hot形式储存## 将每一个 
# 读取文件所有内容
with open('./datasets/train/targets/target.txt','r') as f:
    tars=f.readlines()
# 向量不同位置对应的结果
tar_temp=[0,1,2,3,4,5,6,7,8,9,'.']
labels_train=[]
# 构造one-hot形式的向量集
for i in tars:
    b=np.array([i[0]==str(tar_temp[j]) for j in range(len(tar_temp))])+0
    labels_train.append(b)  # 一个包含所有结果的向量集（与图片集一一对应）


for i in range(len(features_train)):
    cv2.imshow('feature',features_train[i].reshape(28,28))
    print(np.argmax(labels_train[i]))
    cv2.waitKey(500) # 单张图片的显示时间ms

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)
# 定义偏置工厂函数
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)
# 定义卷积矩阵工厂函数
def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义池化层矩阵工厂函数
def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## 主要有两层卷积运算
# 第一层卷积层定义
W_conv1 = weight_variable([5, 5, 1, 32],name='w_conv1')  # 权重变量
b_conv1 = bias_variable([32],name='b_conv1',)            # 偏置

# 图片输入空间生成,结果空间生成（请求组织先分配好茅坑）
x = tf.placeholder("float", shape=[None, 28*28],name="X")  # 
y_ = tf.placeholder("float", shape=[None, 11],name="Y")

# 将输入空间重新塑造为28*28*1（1指单通道，-1是指可以随机应变），为了后面的卷积运算 因为输入是一个向量集
x_image = tf.reshape(x, [-1,28,28,1]) 

# 定义卷积矩阵并计算
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 定义池化层 
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积层定义
W_conv2 = weight_variable([5, 5, 32, 64],name='w_conv2')  # 权重变量
b_conv2 = bias_variable([64],name='b_conv2')              # 偏置

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 经过两次卷积和池化，最后的图片只有7*7了，但是还是不知道他到底是什么鬼，所以再来一个权重矩阵，来算算他到底是什么鬼
# 第一个与处理后图片尺寸一样的权重矩阵变量和偏置变量，直接点乘
W_fc1 = weight_variable([7 * 7 * 64, 1024],name='w_fc1')  
b_fc1 = bias_variable([1024],name='b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob为了防止过拟合，具体原理我还没看到。。。。
keep_prob = tf.placeholder("float",name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 再来一个矩阵使得其变成一个全连接层，所谓全连接层就是一个向量，之所以要将矩阵化为全连接层
# 就是为了使得他通过和再一个权重相乘能够得到和结果维度相同的输出
W_fc2 = weight_variable([1024, 11],name='w_fc2')
b_fc2 = bias_variable([11],name='b_fc2')
# 最后的结果输出为一个向量 和labels相同的维度，
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 设置模型格式 ，添加输出的格式进去
tf.add_to_collection('yconv',y_conv)
saver = tf.train.Saver()

# 训练模型
with tf.Session() as sess:
    # 设置交叉熵为损失函数
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    # 设置优化参数，采用AdamOptimizer优化方法，比最速下降法更优，能够防止过拟合
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 判断预测结果和真实结果是否相同
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # 精度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 初始化各个变量
    sess.run(tf.initialize_all_variables())
    # 迭代训练
    for i in range(201):
        # 随机选取数据进行训练
        sample = random.sample(range(len(labels_train)),42)
        batch_xs=np.array([features_train[i] for i in sample])
        batch_ys=np.array([labels_train[i] for i in sample])
        # 当是100倍数是保存模型，并且输出当前测试精度，保存路径为相对路径
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print ("step %d, training accuracy %g"%(i, train_accuracy))
            save_path = saver.save(sess, "./datasets/digit_model/my_digit_model")
        train_step.run(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 0.5})
#     # 测试整体精度，加载测试集
#     # print ("test accuracy %g"%accuracy.eval(feed_dict={x: features_test, y_: labels_test, keep_prob: 1.0}))