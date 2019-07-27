# -*- coding:utf-8 -*-

import cv2
import numpy as np
'''
# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8), -1)
    return
'''
'''
def cv_imread(file_path = ""):
    file_path_gbk = file_path.encode('gbk')        # unicode转gbk，字符串变为字节数组
    img_mat = cv2.imread(file_path_gbk.decode())  # 字节数组直接转字符串，不解码
    return img_mat
'''
import os
# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    root_dir, file_name = os.path.split(file_path)
    pwd = os.getcwd()
    if root_dir:
        os.chdir(root_dir)
    cv_img = cv2.imread(file_name)
    os.chdir(pwd)
    return cv_img


img = cv2.imread('debug_chineseMat18.jpg')
img1 = cv_imread('./train/charsChinese/京/debug_chineseMat18.jpg') / 255.0
print(type(img1[0][0][0]))

# cvtColor方法: 图片，转换模式: BGR转换为灰度(颜色空间转换)
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) / 255.0    # 颜色空间转换 1 data 2 BGR gray
                                                    # dst.shape = (20, 20)
cv2.waitKey(0)




print('aaa')
img
