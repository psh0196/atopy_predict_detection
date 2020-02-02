### gan output file
### 작성자 : 박서현 
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator, array_to_img
from keras.models import load_model
from PIL import Image
'''
data = glob.glob('/home/psh/autoencoder/atopy/*.jpg')
# print(len(data))
# train_cleaned = glob.glob('/home/psh/train_cleaned/*.png')
# test = glob.glob('/home/psh/test/*.png')
# train,test = train_test_split(data, test_size=0.2,random_state = 0)
X_train = []
for img in data:
	img = load_img(img, grayscale = False, target_size = (224,224,3))
	img = (img_to_array(img).astype('float32')-127.5)/127.5
	X_train.append(img)
# Rescale -1 to 1
X_train = np.array(X_train)
'''
optimizer = Adam(0.0002, 0.5)
model = load_model('dcgan_generator_5000.hdf5')
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizer)
# noise_input = Input(shape=(100,), name="noise_input")
for i in range(32):
  noise_data = np.random.rand(1, 100)
  generated_images = 0.5 * model.predict(noise_data) + 0.5
  print(generated_images.shape)
  generated_images = np.squeeze(generated_images, axis=0)
  print(generated_images.shape)
  generated_images = array_to_img(generated_images)
  print(generated_images.size)
  generated_images.save('/home/psh/autoencoder/gan/'+str(i)+'.jpg')

