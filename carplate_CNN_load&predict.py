# -*- coding:utf-8 -*-

from keras.models import load_model
from keras.models import Model
import numpy as np
import cv2


# 加载模型
model = load_model('carplate_CNN_model.h5')

# 加载一张图片
img = cv2.imread('./13-4.jpg')    # 一张 20*20 的图片
# 转换模型所需要的输入格式 单通道灰度  float64 型
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

# 模型需要输入一个四维的数据，所以要把二维的 20*20 的图片  转换为 1*1*20*20 的格式输入模型以预测
temp: np.ndarray = np.ndarray(shape=(1, 1, 20, 20), dtype='float64')
temp[0][0] = img
ret: np.ndarray = model.predict(temp)
# ret 是一个 shape 为(1, 65) 的 ndarray, 不为 0 的那个位置即为类型序号

# 标签位置顺序
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
# 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
# '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘',
# '皖','粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']

# 在同个位置找到标签输出即可
(_, index) = np.where(ret == ret.max())
import label_define   # 训练时的标签定义

result = label_define.car_label[index[0]]    # result: str
print(result)

