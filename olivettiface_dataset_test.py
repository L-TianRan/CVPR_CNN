from PIL import Image
import numpy as np
import pickle
'''
img = Image.open('.\\train\\chars2\\0\\4-3.jpg')
print(img)
'''


img = Image.open('.\\olivettifaces.gif')
print(img)
print(img.size)

img_ndarray = np.asarray(img, dtype='float64') / 255
print(img_ndarray)
print(img_ndarray.size)
print(len(img_ndarray))
print(len(img_ndarray[0]))

# create numpy array of 400*2679
img_rows, img_cols = 57, 47
face_data = np.empty((400, img_rows*img_cols))
# convert 1140*942 ndarray to 400*2679 matrix

for row in range(20):
    for col in range(20):
        face_data[row*20+col] = np.ndarray.flatten(img_ndarray[row*img_rows:(row+1)*img_rows, col*img_cols:(col+1)*img_cols])
# numpy.ndarray.flatten函数的功能是将一个矩阵平铺为向量

print(len(face_data)) # = 400
print(len(face_data[0])) # = 2679 = 57*47

# create label
face_label = np.empty(400, dtype=int)
for i in range(400):
    face_label[i] = i / 10

# pickling file
f = open('.\\olivettifaces.pkl', 'wb')
# store data and label as a tuple
pickle.dump((face_data,face_label), f)
f.close()