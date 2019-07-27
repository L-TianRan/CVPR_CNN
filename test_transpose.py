import numpy as np

data = []
label = []

for i in range(10):
    temp = np.array([i] * 5)   # temp is ndarray

    data.append(np.ndarray.flatten(temp))
    label.append(i)

temp_2d = np.array([data, label])

# 转置函数
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


temp_2d_t = transpose(temp_2d)
tem_2d =None
