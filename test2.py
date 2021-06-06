
# -*- encoding: utf-8 -*-
'''
@File    :   test2.py
@Time    :   2021/06/06 17:09:03
@Author  :   ZH_Somin 
@Version :   1.0
@Contact :   achouloves@163.com
'''

# here put the import lib

import cv2
import numpy as np
import tensorflow as tf
import tensorflow as tf
import paho.mqtt.client as mqtt
import time
import hashlib
import hmac



# 不支持多行数字识别，以及单行多个小数点的数值识别（单行只能实现字符串识别），
# labels的各个位置代表的数字

tar_temp=[0,1,2,3,4,5,6,7,8,9,'.'] 
# 定义一个阈值函数，将数码管部分取出来，根据实际情况进行相应修改，找到最优参数
def thresholding_inv(image):
    kernel = np.ones((3,3),np.uint8) 
    # 定义膨胀核心，根据实际情况进行修改
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 6))# 1代表横向膨胀，6代表纵向膨胀
    ## 腐蚀参数我已经注释掉，根据实际情况选择是否使用
    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3)) 
    ## 根据RGB图得到灰度图
    blur = cv2.blur(image,(1,3))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # 灰度图二值化
    ret, bin = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    bin = cv2.erode(bin,kernel,iterations = 1)## 对灰度图进行腐蚀，主要是为了分离相近的小数点，如果足够清晰可以不使用腐蚀，我已注释掉
    # bin = cv2.erode(bin,kernel_erode)
    ## 对灰度图进行膨胀
    bin=cv2.dilate(bin,kernel_dilate,iterations = 2)
    
    return bin

# Read the input image
## demo 图像在此目录下
if __name__ == '__main__':
    im = cv2.imread('./image/01.jpg')  # 还有 1-6 张图 修改最后一个数即可
    print(im.shape)
    # im = im[500:1800,300:1800]
    ## 二值化处理
    im_th = thresholding_inv(im)

    # 显示图片
    # cv2.imshow('im_th',im_th)
    # cv2.waitKey(0) # 显示1000ms 

    # Find contours in the image  寻找边界集合
    _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get rectangles contains each contour 
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    # 加载训练好的模型，并预测通过
    with tf.Session() as sess:
        # 加载模型的结构框架graph
        new_saver = tf.train.import_meta_graph('./datasets/digit_model/my_digit_model.meta')
        # 加载各种变量
        new_saver.restore(sess,'./datasets/digit_model/my_digit_model')
        yy_hyp = tf.get_collection('yconv')[0]
        graph = tf.get_default_graph() 
        X = graph.get_operation_by_name('X').outputs[0]#为了将 x placeholder加载出来
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0] # 将keep_prob placeholder加载出来
        # mm用来保存数字以及数字坐标
        mm={}
        # for循环对每一个contour 进行预测和求解，并储存
        for rect in rects:
            # Draw the rectangles 得到数字区域 roi
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            # Make the rectangular region around the digit
            leng1= int(rect[3])
            leng2= int(rect[2])
            pt1 = int(rect[1] )
            pt2 = int(rect[0] )
            # 得到数字区域
            roi = im_th[pt1:pt1+leng1, pt2:pt2+leng2]
            # 尺寸缩放为模型尺寸
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            # 处理成一个向量，为了和模型输入一直
            roi=np.array([roi.reshape(28*28)/255])
            # 运行模型得到预测结果
            pred= sess.run(yy_hyp,feed_dict = {X:roi,keep_prob:1.0})
            # 得到最大可能值索引 ind
            ind=np.argmax(pred)
            #labels不同位置代表的不同数字   (tar_temp[ind]) 就是预测值
            # 将预测值添加到图像中，并显示
            cv2.putText(im, str(tar_temp[ind]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            # 储存每个数字和其对应的boundingbox的像素点坐标
            mm[pt2]=tar_temp[ind]
        # 最后的处理
        # 根据像素坐标，从左到右排序，得到数字的顺序
        num_tup=sorted(mm.items(),key=lambda x:x[0])
        # 将数字列表连接为字符串
        num=(''.join([str(i[1]) for i in num_tup]))
        try:
            numn=float(num)
            print('图中数字为%s,数值大小为%s' %(num,numn))
        except:
            print('不好意思，目前不支持多个小数点的数值识别')
            print('图中数字为%s'% num)
        # 显示图像 
        cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
        cv2.imshow("Resulting Image with Rectangular ROIs", im)
        cv2.waitKey(0)