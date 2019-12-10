### 전체 데이터들을 나누기 위해 소수 데이터들을 학습하는 파일 
### 작성자 : 박서현 

import numpy as np 
import pandas as pd
import os
import keras
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
import sys
import os 
import gc 
# import bcolz 

from random import randint 
import numpy as np 
import pandas as pd 


import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
# import seaborn as sns


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
from keras.models import Model
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet
from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50



from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D,Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
# import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
'''
path_dir = '/home/psh/final'
label = []
file_list = os.listdir(path_dir)
print(file_list)
for item in file_list:
	if 'skin' in item:
		label.append(0)
	elif 'atopy' in item:
		label.append(1)
	elif 'background' in item:
		label.append(2)
# print(label)

import glob
images = glob.glob("/home/psh/final/*.jpg")
X = []

for i in images :
    X.append((cv2.imread(i)))
norm = []
for i in range(len(X)) :
    norm.append(cv2.resize(src=X[i], dsize=(224, 224), interpolation=cv2.INTER_AREA)/255)
X = norm

nb_classes = int(np.array(label).max()+1)
nb_classes
y = np_utils.to_categorical(label, nb_classes)
# print(X)
# print(y)
# print(norm)
'''

# np.save('X.npy',X)
# np.save('y.npy',y)

X = np.load('X.npy')
y = np.load('y.npy')

# df = pd.read_csv('label_check.csv')
'''
X = []
for i in range(len(X)) :
    X.append((cv2.imread(i)))
norm = []
'''
'''
for i in range(len(X)) :
    norm.append(cv2.resize(src = X[i], dsize = (224, 224), interpolation = cv2.INTER_AREA)/255)
X = np.array(norm)

nb_classes = int(np.array(y).max()+1)
y = np_utils.to_categorical(y, nb_classes)
'''
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, shuffle=True)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

img_rows, img_cols, img_channel = 224, 224, 3





model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation="sigmoid"))
from keras.utils import multi_gpu_model
model = multi_gpu_model(model,gpus=2)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.0001),
#               metrics=['accuracy'])

history = model.fit(train_x, train_y, batch_size=128,
                    epochs=50, validation_split=0.3,
                    verbose=1)


print('Testing...')
score = model.evaluate(test_x, test_y,
                       batch_size=128, verbose=1)
# print("\nTest score:", score[0])
print('Test accuracy:', score[1])

fig = plt.figure()
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
plt.savefig("graph.png", transparent=True)

from keras.models import load_model
model.save('keras_model.h5')
model.save_weights('model_weight.h5')
import glob
images = glob.glob("/home/psh/final/*.jpg")
X = []

for i in images :
    X.append((cv2.imread(i)))
norm = []
for i in range(len(X)) :
    norm.append(cv2.resize(src = X[i], dsize = (224, 224), interpolation = cv2.INTER_AREA)/255)
X = np.array(norm)
file_list = os.listdir('/home/psh/final')
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
