#### resnet test
#### 작성자 : 박서현 

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
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
from keras.initializers import glorot_uniform
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split

X = []
df = pd.read_csv('final.csv')

for i in range(len(df)) :
    img = (cv2.imread("/home/psh/pre_all/"+df["name"][i])).astype(float)
    X.append(img)
    
norm = []
for i in range(len(X)) :
    img = (cv2.resize(src = X[i], dsize = (224, 224),interpolation=cv2.INTER_AREA)/255.0)
    norm.append(img)
X = norm


nb_classes = int(np.array(df["label"]).max()+1)
y = list(np_utils.to_categorical(df["label"], nb_classes))

np.save("X.npy",X)
np.save("y.npy",y)

X = np.load("X.npy")
y = np.load("y.npy")

y= y[:,np.newaxis,np.newaxis,:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

IMG_ROWS = 224
IMG_COLS = 224
IMG_CHANNELS = 3

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (?3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X =  Activation('relu')(X)

    # Third component of main path (?2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (?2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
    

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s),  kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (?3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path (?2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), kernel_initializer = glorot_uniform(seed=0))(X)
    X =  BatchNormalization(axis = 3)(X)

    ##### SHORTCUT PATH #### (?2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding='valid',
     kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (?2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
    
    
def ResNet50(input_shape = (224, 224, 3), classes = 3):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2),  kernel_initializer = glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis = 3, )(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (?4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 2, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=2, block='d')

    # Stage 4 (?6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 2, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=2, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=2, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=2, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=2, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=2, block='f')

    # Stage 5 (?3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 2, block='a', s = 2)
    X =identity_block(X, 3, [512, 512, 2048], stage=2, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=2, block='c')

    X = Conv2D(2048, (3, 3), strides = (1,1))(X)
    # X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(2048, (3, 3), strides = (1,1))(X)
    # X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    
    # AVGPOOL (?1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name = 'avg_pool')(X)
    
    ### END CODE HERE ###

    # output layer
    # X = Flatten()(X)
    X = Conv2D(3,(1,1), activation='softmax')(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
  
model = ResNet50(input_shape = (224, 224, 3), classes = 3)

 
model.summary()

from keras.utils import multi_gpu_model
# model = multi_gpu_model(model,gpus=2)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32,
                    epochs=50, validation_split=0.3,
                    verbose=1)

print('Testing...')
score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print('Test accuracy:', score[1])

fig = plt.figure()
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
fig.savefig('/home/psh/batch_graph.png')

from keras.models import load_model
model.save('batch_model.h5')
model.save_weights('batch_weight.h5')
