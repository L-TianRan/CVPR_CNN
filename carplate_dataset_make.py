from typing import List, Any

from PIL import Image
import numpy as np
import pickle
import os
import math
import cv2 as cv
import random

rootPath = './train'
charPath = '/chars2'
chinese_charPath = '/charsChinese'
chars_set = os.listdir(rootPath + charPath)
# charsfiles = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
# 'A', 'B', 'C', 'D', 'E', 'F', 'G',
# 'H', 'J', 'K', 'L', 'M', 'N', 'P',
# 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
# 'X', 'Y', 'Z']

for chinese in os.listdir((rootPath + chinese_charPath)):
    chars_set.append(chinese)

#  添加省名
#  '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼',
#  '甘', '皖', '粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']

print(chars_set)    # 共65类
input("正在制作数据集，是否继续？ y/n")
chars_data = []    # 训练集
chars_label = []

test_data = []    # 测试集
test_label = []

img_lenth = 20    # 数据集长宽均为20个像素
img_width = 20


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    root_dir, file_name = os.path.split(file_path)
    pwd = os.getcwd()
    if root_dir:
        os.chdir(root_dir)
    cv_img = cv.imread(file_name)
    os.chdir(pwd)
    return cv_img


for chars in chars_set:

    if os.path.exists(rootPath + charPath + '/' + chars):    # 数字部分成立
        prePath = rootPath + charPath
    else:  # 汉字部分
        prePath = rootPath + chinese_charPath

    char_image_name_set = os.listdir(prePath + '/' + chars)    # 目录下的文件名集合

    for image_name in char_image_name_set:      # 获得文件名
        image_path = prePath + '/' + chars + '/' +image_name   # 获得文件相对路径名

        if not os.path.isdir(image_path):
            img = Image.open(image_path)    # 打开图像文件
            img_ndarray: np.ndarray = np.asarray(img, dtype='float64') / 255    # 转为数组
            if img_ndarray.shape == (20, 20):   # 发现数据集里有非灰度图像 调试后发现其形状均为 (20, 20, 3)
                chars_data.append(np.ndarray.flatten(img_ndarray))     # 把矩阵转为向量, 并加入数据集中
                chars_label.append(chars_set.index(chars))     # 添加标签 标签的int值为字符在chars_set中的序号
            elif img_ndarray.shape == (20, 20, 3):
                temp = cv_imread(image_path)
                img_ndarray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY) / 255.0    # 转换成单通道 现在img_ndarray.shape == (20, 20)
                # 正常添加
                chars_data.append(np.ndarray.flatten(img_ndarray))     # 把矩阵转为向量, 并加入数据集中
                chars_label.append(chars_set.index(chars))     # 添加标签 标签的int值为字符在chars_set中的序号



            # print(len(chars_data))
            # print(len(chars_data[0])) = 400 = 20 * 20


# 字符图像数据集在 chars_data 列表中
# 对应的标签在 chars_label 列表中
# 都是按序存放的 现在将列表随机打乱
temp = []
temp = np.array([chars_data, chars_label])   # 转换成2 维矩阵  第一维是图像列表  第二维是图像对应标签

def transpose(set_2d: np.ndarray):    # 指定参数类型

    img = np.ndarray(shape=(400, ))
    label = np.ndarray(shape=(set_2d.shape[1], ))
    ret: np.ndarray = np.array([img, label])   # 初始化形状
    ret: list = []     # 每行存放一张图片 一个标签
    if set_2d.shape[0] == 2:   # 参数格式符合要求
        for i in range(set_2d.shape[1]):    # shape[1] 是数据个数
                                            # set_2d[0][i] 是一张图片, len(set_2d[0][i]) 是该图像像素个数
                                            # set_2d[1][i]] 是对应标签
            ret.append(np.array([set_2d[0][i], set_2d[1][i]]))
    ret = np.array(ret)
    return ret

temp = transpose(temp)    # 转置  使图像和对应标签在同一行
np.random.shuffle(temp)    # 按行随机打乱顺序
np.random.shuffle(temp)    # 再打乱一次

all_chars_data = list(temp[:, 0])    # 第 0 列   是打乱后的图像列表
all_chars_lable = list(temp[:, 1])    # 第 1 列  是对应的标签列表


# 抽取一部分当测试集  剩下的是训练集
total_img_num = len(all_chars_lable)    # 总共图片的个数
test_img_rate = 0.01     # 总共约有16000 张图片  取其中的0.01 用来当作测试图片
test_img_num = math.ceil(total_img_num * test_img_rate)     # 要取的测试图片数目

# 直接从打乱后的数据集里取前 N 项当作测试集
test_data = all_chars_data[:test_img_num]
test_label = all_chars_lable[:test_img_num]
# 剩下的就是训练集
train_data = all_chars_data[test_img_num:]
train_label = all_chars_lable[test_img_num:]

# 把list转为ndarray
train_label = np.array(train_label)
train_data = np.array(train_data)

test_label = np.array(test_label)
test_data = np.array(test_data)

input("生成文件 是否继续 y/n ?")

# pickling file
# 生成训练集
f = open('./carplate_trainset.pkl', 'wb')
# store data and label as a tuple
pickle.dump((train_data, train_label), f)
f.close()

# 生成测试集
f = open('./carplate_testset.pkl', 'wb')
# store data and label as a tuple
pickle.dump((test_data, test_label), f)
f.close()
