### vgg test
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

X = []
df = pd.read_csv('final.csv')
for i in range(len(df)) :
  print('/home/psh/pre_all/'+df["name"][i])
  img = (cv2.imread("/home/psh/pre_all/"+df["name"][i])).astype(float)
  X.append(img)
# print(X)
norm = []
for i in range(len(X)) :
  img = (cv2.resize(src = X[i], dsize = (224, 224),interpolation=cv2.INTER_AREA)/255.0)
  
  norm.append(img)
X = norm
# print(X)
# print(X[0].dtype)

nb_classes = int(np.array(df["label"]).max()+1)
y = list(np_utils.to_categorical(df["label"], nb_classes))

np.save("yolo_X.npy",X)
np.save("yolo_y.npy",y)

X = np.load("yolo_X.npy")
y = np.load("yolo_y.npy")
y = y[:,np.newaxis,np.newaxis,:]
print(y.shape)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=42)
print(len(train_x))
print(len(test_x))
IMG_ROWS = 224
IMG_COLS = 224 
IMG_CHANNELS = 3
model = Sequential()
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

model.summary()
from keras.utils import multi_gpu_model
model = multi_gpu_model(model,gpus=2)

# model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

history = model.fit(train_x, train_y, batch_size=128,
                    epochs=50, validation_split=0.3,
                    verbose=1)

print('Testing...')
score = model.evaluate(test_x, test_y,
                       batch_size=128, verbose=1)

print('Test accuracy:', score[1])
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

from keras.models import load_model
model.save('yolo_model.h5')
model.save_weights('yolo_weight.h5')

import glob
images = glob.glob("/home/psh/pre_all/*.jpg")
X = []

for i in images :
    X.append(cv2.imread(i).astype(float))
norm = []
for i in range(len(X)) :
    norm.append(cv2.resize(src = X[i], dsize = (224, 224), interpolation = cv2.INTER_AREA)/255.0)
X = np.array(norm)
file_list = os.listdir('/home/psh/pre_all')
image_name = []
for item in file_list:
	image_name.append(item)

prediction = model.predict(X)
prediction = np.argmax(prediction,axis=3)
prediction = np.ravel(prediction, order='C')
print(prediction)
print("y: ",prediction)
# np.save('prediction.npy',prediction)
prediction_df =pd.DataFrame({"image_name":image_name,"label":prediction})
print(prediction_df)
prediction_df.to_csv("test.csv")
