## yolo_test 실제 예측 파일(model 아키텍처 부분이랑 이미지 사이즈 부분 수정해서 써야함)
## 작성자 : 박서현 
import pandas as pd
import cv2
import os
import sys
import gc
from random import randint
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras import applications
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras import optimizers
from keras.models import load_model
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import h5py
# from h5py_loader import *
images = glob.glob("/home/psh/test_img/*.JPG")
X = []

for i in images :
  X.append(cv2.imread(i).astype(float))

# print(X[0].shape)
norm = []
for i in range(len(X)) :
    norm.append(cv2.resize(src = X[i], dsize = (6016, 4016), interpolation = cv2.INTER_AREA)/255.0)
    # norm.append(X[i]/255.0)
X = np.array(norm)
# X = X[np.newaxis,:,:,:]
# print(X)
IMG_ROWS=4016
IMG_COLS=6016
IMG_CHANNELS=3


### ====================== best model modify ============================
model = Sequential()
model1 = load_model('yolo_model.h5')


model.add(Conv2D(64, (3, 3),padding='same',input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding='valid'))

model.add(Conv2D(128, (3, 3),padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding='valid'))


model.add(Conv2D(256, (3, 3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding='valid'))


model.add(Conv2D(256, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding='valid'))


model.add(Conv2D(512, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding='valid'))


model.add(Conv2D(512, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (1,1),padding='valid'))


model.add(Conv2D(512, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding='valid'))


model.add(Conv2D(1024, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (1,1),padding='valid'))


model.add(Conv2D(1024, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (1,1),padding='valid'))


model.add(Conv2D(4096, (1, 1),padding='same'))


model.add(Conv2D(1000, (1, 1),padding='same'))


model.add(Conv2D(3, (1, 1), activation='softmax'))

model.set_weights(model1.get_weights())

model.summary()
## ============================================================
# model.save('yolotest_model.h5')

# model.load_weights('yolo_weight.h5')
from keras.utils import multi_gpu_model
model = multi_gpu_model(model,gpus=2)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
image_size = (4016,6016)

## find tensor location
def tensor_calculator(pixel):
  x = pixel[0]
  y = pixel[1]
  for i in range(5):
    x = int(x/2)
    y = int(y/2)
  x = int((x-1)/2)-2
  y = int((y-1)/2)-2
  return(int(x),int(y))
tensor = tensor_calculator(image_size)
print(tensor)

step_w = 224-math.ceil(float(224*tensor[1]-image_size[1])/float(tensor[1]-1))
step_h = 224-math.ceil(float(224*tensor[0]-image_size[0])/float(tensor[0]-1))
print(step_w)
print(step_h)
# print(step_w)
# print(step_h)


'''
imglist = os.listdir('/home/psh/ori/')
print(len(X))
print(len(imglist))
'''
skin_pred = np.zeros((1,tensor[0],tensor[1]))
imgarr = np.zeros((1,image_size[0],image_size[1],3), dtype=np.float32)
for i in range(len(images)):
  img = Image.open(images[i])
  imgarr[0] = np.asarray(img, dtype=np.float32)/255
  skin_pred[0] = np.argmax(model.predict(imgarr),axis=-1)
  
  fig, ax = plt.subplots(1,figsize=(image_size[1]/60, image_size[0]/60),dpi=60)
  ax.imshow(img)
  
  # prediction = np.argmax(prediction,axis=3)
  # prediction = np.ravel(prediction, order='C')
  # df1 = pd.DataFrame(prediction[0])
  # df1.to_csv('/home/psh/skin_result/'+str(i)+'.csv')
  # prediction = prediction.reshape(60,91,order='C')
  print(skin_pred[0])
  for j in range(tensor[0]):
    for k in range(tensor[1]):
      if skin_pred[0][j][k]==1:
      # if j == 30 and k == 45:
        ## detection
        rect = patches.Rectangle((k*step_w,j*step_h),224,224,edgecolor='blue',facecolor='none',alpha=1)
        ax.add_patch(rect)
  plt.savefig('/home/psh/test_result/'+str(i)+'.jpg')


