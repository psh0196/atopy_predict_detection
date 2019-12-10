### 소수 데이터들을 예축한 케라스 모델을 이용하여 전체 데이터를 구분하여 테이블을 만든 파일 
### 작성자 : 박서현 
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
model = load_model('keras_model.h5')


import glob
images = glob.glob("/home/psh/40000/*.jpg")
X = []

for i in images :
    X.append(cv2.imread(i).astype(float))
norm = []
for i in range(len(X)) :
    norm.append(cv2.resize(src = X[i], dsize = (224, 224), interpolation = cv2.INTER_AREA)/255.0)
X = np.array(norm)
file_list = os.listdir('/home/psh/40000')
image_name = []
for item in file_list:
	image_name.append(item)

prediction = model.predict(X)

prediction = np.argmax(prediction,axis=1)
print(prediction)
print("y: ",prediction)
# np.save('prediction.npy',prediction)
prediction_df =pd.DataFrame({"image_name":image_name,"label":prediction})
print(prediction_df)
prediction_df.to_csv("final.csv")

