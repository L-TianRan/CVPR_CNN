from PIL import Image
import numpy as np
import pickle
import os
import keras
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# 标签位置顺序
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
# 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
# '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘',
# '皖','粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']

img_height = 20    # 图片形状为 20 * 20
img_width = 20

# 模型输入一张20*20 的灰度图像
# 输出上述标签的一个, onehot格式

# load carplate_trainset
f = open('.\\carplate_trainset.pkl', 'rb')
train_data, train_label = pickle.load(f)
f.close()
# load carplate_testset
f = open('.\\carplate_testset.pkl', 'rb')
test_data, test_label = pickle.load(f)
f.close()

print(len(train_data))    # the number of train_data  16231
print(len(train_data[0]))
print(len(test_data))   # the number of test_data    164
print(len(test_data)/len(train_label))

input('开始训练模型  任意键继续')
# chars_label_onehot = keras.utils.to_categorical(chars_label, 65)    # 65类  包括汉字 数字 字母

# print(chars_label_onehot)


np.random.seed(1337) #for reproducibility再现性


def reshape(data: np.ndarray):    # data 是平铺后的图片数组
# 把原来平铺的图片像素展开成 20 * 20
        num_data = len(data)    # 图片个数
        ret = []   # 要把data平铺后的再展开
        for i in range(num_data):
                if len(data[i]) == 400:
                        temp = data[i].reshape(img_height, img_width)
                else:
                        temp = data[i]
                ret.append(temp)
        ret = np.array(ret)
        return ret


# data pre-processing
x_train = reshape(train_data)  # 把展平的图片恢复
x_test = reshape(test_data)

x_train = x_train.reshape(-1, 1, 20, 20)   # -1代表个数不限，1为高度，黑白照片高度为1
x_test = x_test.reshape(-1, 1, 20, 20)
y_train = np_utils.to_categorical(train_label, num_classes=65)     # 把标签变为65个长度，若为1，则在1 处为1，剩下的都标为0
y_test = np_utils.to_categorical(test_label, num_classes=65)       # 即改为onehot格式


# 写一个LossHistory类，保存loss和acc
# https://blog.csdn.net/u013381011/article/details/78911848


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

# Another way to build CNN
model = Sequential()

# Conv layer 1 output shape (32,20,20)
model.add(Convolution2D(
        nb_filter =32,     # 滤波器装了32个，每个滤波器都会扫过这个图片，会得到另外一整张图片，所以之后得到的告诉是32层
        nb_row=5,
        nb_col=5,
        border_mode='same',   # padding method
        input_shape=(1,       # channels  通道数
                     20, 20),    # 图片 height & width 长和宽
        ))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32,10,10)
model.add(MaxPooling2D(
        pool_size=(2, 2),   # 2*2
        strides=(2, 2),    # 长和宽都跳两个再pool一次
        border_mode='same',   # paddingmethod
        ))

# Conv layers 2 output shape (64,10,10)
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))

# Pooling layers 2 (max pooling) output shape (64,5,5)
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

# Fully connected layer 1 input shape (64*5*5) = (1600)
# Flatten 把三维抹成一维，全连接
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (65) for 65 classes
model.add(Dense(65))   # 输出65个单位
model.add(Activation('softmax'))   # softmax用来分类

#Another way to define optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(   # 编译
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'],    # 在更新时同时计算一下accuracy
        )
# create an instance of LossHistory
history = LossHistory()

print("Training~~~~~~~~")
# Another way to train the model
model.fit(x_train, y_train,
          epochs=50, batch_size=32,    # 训练50批，每批32个
          callbacks=[history]     # 加入callback 计算loss
          )

print("\nTesting~~~~~~~~~~")
# Evalute the model with the  metrics we define earlier 评估精确度
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss:',loss)
print('\ntest accuracy:', accuracy)

# 绘制acc-loss曲线
history.loss_plot('epoch')





# epoch = 50 batchsize = 32
# 训练准确度99.72%左右
# 测试集中准确度 96.34%左右
input("wait for save")
model.save('carplate_CNN_model.h5')
