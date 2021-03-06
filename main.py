# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/06/06 17:08:33
@Author  :   ZH_Somin 
@Version :   1.0
@Contact :   achouloves@163.com
'''

# here put the import lib

# -*- coding: UTF-8 -*-

import cv2
import os
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import time
import json
import hashlib
import hmac
from PIL import Image


#在这里面需要加入阿里云的三元组相关信息
options = {
    'productKey':'a17FMRIYU5o',
    'deviceName':'T001',
    'deviceSecret':'fb360283b4a80d71e39fe91194d4c19b',
    'regionId':'cn-shanghai'
}            

HOST = options['productKey'] + '.iot-as-mqtt.'+options['regionId']+'.aliyuncs.com'
PORT = 1883 
PUB_TOPIC = "/sys/" + options['productKey'] + "/" + options['deviceName'] + "/thing/event/property/post";

###连接阿里云需要的几个函数
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # client.subscribe("the/topic")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

def hmacsha1(key, msg):
    return hmac.new(key.encode(), msg.encode(), hashlib.sha1).hexdigest()

def getAliyunIoTClient():
	timestamp = str(int(time.time()))
	CLIENT_ID = "paho.py|securemode=3,signmethod=hmacsha1,timestamp="+timestamp+"|"
	CONTENT_STR_FORMAT = "clientIdpaho.pydeviceName"+options['deviceName']+"productKey"+options['productKey']+"timestamp"+timestamp
	# set username/password.
	USER_NAME = options['deviceName']+"&"+options['productKey']
	PWD = hmacsha1(options['deviceSecret'],CONTENT_STR_FORMAT)
	client = mqtt.Client(client_id=CLIENT_ID, clean_session=False)
	client.username_pw_set(USER_NAME, PWD)
	return client

#识别需要用到的数据
tar_temp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '.']


# 定义一个阈值函数，将数码管部分取出来，这里属于对采集到的图片进行预处理，先进行平滑滤波，进行二值化操作，对图像进行形态学的操作 包括
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
while True:
    if __name__ == '__main__':
        
        client = getAliyunIoTClient()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(HOST, 1883, 300)

        img = Image.open('./image/112.jpg')
        size_img = img.size
        print(size_img)
        # 准备将图片切割成4张小图片,这里后面的2是开根号以后的数，比如你想分割为9张，将2改为3即可     这里感谢师弟！
        weight = int(size_img[0] // 2)
        height = int(size_img[1] // 2)
        names = locals()
        for j in range(2):
            for k in range(2):
                box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
                region = img.crop(box)
                # 输出路径
                region.save('image/{}{}.jpg'.format(j, k))
        for j in range(2):
            for k in range(2):
                im = cv2.imread('image/{}{}.jpg'.format(j, k))  # 还有 1-6 张图 修改最后一个数即可
                ## 二值化处理
                im_th = thresholding_inv(im)
        
                # 显示图片
                # cv2.imshow('im_th',im_th)
                # cv2.waitKey(1000) # 显示1000ms

                # Find contours in the image  寻找边界集合
                _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Get rectangles contains each contour
                rects = [cv2.boundingRect(ctr) for ctr in ctrs]
                # 加载训练好的模型，并预测通过
                with tf.Session() as sess:
                    # 加载模型的结构框架graph
                    new_saver = tf.train.import_meta_graph('./datasets/digit_model/my_digit_model.meta')
                    # 加载各种变量
                    new_saver.restore(sess, './datasets/digit_model/my_digit_model')
                    yy_hyp = tf.get_collection('yconv')[0]
                    graph = tf.get_default_graph()
                    X = graph.get_operation_by_name('X').outputs[0]  # 为了将 x placeholder加载出来
                    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]  # 将keep_prob placeholder加载出来
                    # mm用来保存数字以及数字坐标
                    mm = {}
                    # for循环对每一个contour 进行预测和求解，并储存
                    for rect in rects:
                        # Draw the rectangles 得到数字区域 roi
                        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
                        # Make the rectangular region around the digit
                        leng1 = int(rect[3])
                        leng2 = int(rect[2])
                        pt1 = int(rect[1])
                        pt2 = int(rect[0])
                        # 得到数字区域
                        roi = im_th[pt1:pt1 + leng1, pt2:pt2 + leng2]
                        # 尺寸缩放为模型尺寸
                        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                        # 处理成一个向量，为了和模型输入一直
                        roi = np.array([roi.reshape(28 * 28) / 255])
                        # 运行模型得到预测结果
                        pred = sess.run(yy_hyp, feed_dict={X: roi, keep_prob: 1.0})
                        # 得到最大可能值索引 ind
                        ind = np.argmax(pred)
                        # labels不同位置代表的不同数字   (tar_temp[ind]) 就是预测值
                        # 将预测值添加到图像中，并显示
                        cv2.putText(im, str(tar_temp[ind]), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
                        # 储存每个数字和其对应的boundingbox的像素点坐标
                        mm[pt2] = tar_temp[ind]
                    # 最后的处理
                    # 根据像素坐标，从左到右排序，得到数字的顺序
                    num_tup = sorted(mm.items(), key=lambda x: x[0])
                    # 将数字列表连接为字符串
                    a =  names['num' + str(j)+str(k) ] =(''.join([str(i[1]) for i in num_tup]))
                    # print(b)
                    # try:
                    #     
                    #     print('图中数字为%s,数值大小为%s' % (a,b))
                    # except:
                    #     print('不好意思，目前不支持多个小数点的数值识别')
                    #     print('图中数字为%s' % b)
                    # time.sleep(5)
                    # 显示图像
                    # cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
                    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
                    # cv2.waitKey(0)
        payload_json = {
		'id': int(time.time()),
		'params': {
			'V11': float(num00),
            'V12': float(num01),
            'A11': float(num10),
            'A12': float(num11)
		    },
	    'method': "thing.event.property.post"
	    }
        print('send data to iot server: ' + str(payload_json))
        client.publish(PUB_TOPIC,payload=str(payload_json),qos=1)
        time.sleep(5)
client.loop_forever()