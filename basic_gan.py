## 슈퍼컴퓨터 쓰기 전 gan
## 작성자 : 박서현 
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from keras.utils import multi_gpu_model
import shutil
import os

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img


generator_ = Sequential()
generator_.add(Dense(256*7*7,input_dim=100))
generator_.add(Reshape((7,7,256)))

generator_.add(Conv2DTranspose(1024, 4, strides=1, padding='same'))
generator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
generator_.add(ReLU())
    
generator_.add(Conv2DTranspose(512, 4, strides=2, padding='same'))
generator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
generator_.add(ReLU())

    
generator_.add(Conv2DTranspose(256, 4, strides=2, padding='same'))
generator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
generator_.add(ReLU())

generator_.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
generator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
generator_.add(ReLU())
    
generator_.add(Conv2DTranspose(64, 4, strides=2, padding='same'))
generator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
generator_.add(ReLU())

generator_.add(Conv2DTranspose(32, 4, strides=2, padding='same'))
generator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
generator_.add(ReLU())

    
generator_.add(Conv2DTranspose(3, 3, strides=1, activation='tanh', padding='same'))


noise_input = Input(shape=(100,), name="noise_input")
generator = Model(noise_input, generator_(noise_input), name="generator")

generator_.summary()

generator.summary()
# generator = multi_gpu_model(generator, gpus=2)
optimizer = Adam(0.0005, 0.5)

generator.compile(loss='binary_crossentropy', optimizer=optimizer)

noise_data = np.random.normal(0, 1, (32, 100))
generated_images = 0.5 * generator.predict(noise_data) + 0.5
generated_images.shape

def show_images(generated_images, n=4, m=8, figsize=(9, 5)):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
    for i in range(n):
        for j in range(m):
            k = i * m + j
            ax = fig.add_subplot(n, m, i * m + j + 1)
            ax.imshow(generated_images[k][:, :, :])
            ax.grid(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    plt.tight_layout()
    plt.show()

show_images(generated_images)

discriminator_ = Sequential()
discriminator_.add(Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=(224,224,3)))
discriminator_.add(LeakyReLU(0.2))
    
discriminator_.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
discriminator_.add(LeakyReLU(0.2))
    
discriminator_.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
discriminator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
discriminator_.add(LeakyReLU(0.2))
    
discriminator_.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
discriminator_.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
discriminator_.add(LeakyReLU(0.2))
    
discriminator_.add(Conv2D(3, kernel_size=4, strides=1, padding='same'))

discriminator_.add(Flatten())
discriminator_.add(Dense(1, activation='sigmoid'))

image_input = Input(shape=(224, 224, 3), name="image_input")

discriminator = Model(image_input, discriminator_(image_input), name="discriminator")

discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

discriminator_.summary()

discriminator.summary()

noise_input2 = Input(shape=(100,), name="noise_input2")
combined = Model(noise_input2, discriminator(generator(noise_input2)))

combined.summary()

combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
# X_train = np.expand_dims(X_train, axis=3)

batch_size = 128
half_batch = int(batch_size / 2)

def train(epochs, print_step=10):
    history = []
    for epoch in range(epochs):

        # discriminator training
        #######################################################################3
        
        real_images = X_train[np.random.randint(0, X_train.shape[0], half_batch)]
        y_real = np.ones((half_batch, 1))
        generated_images = generator.predict(np.random.normal(0, 1, (half_batch, 100)))
        y_generated = np.zeros((half_batch, 1))
        
       
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, y_real)
        d_loss_fake = discriminator.train_on_batch(generated_images, y_generated)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # generator training
        #######################################################################3
       
        noise = np.random.normal(0, 1, (batch_size, 100))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
        print(g_loss)
        print(d_loss)
        record = (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], 100 * g_loss[1])
        history.append(record)
        if epoch % print_step == 0:
            print("%5d [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f, acc.: %.2f%%]" % record)

history100 = train(100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

from keras.models import load_model

def save_models(epoch):
    generator.save("dcgan_generator_{}.hdf5".format(epoch))
    discriminator.save("dcgan_discriminator_{}.hdf5".format(epoch))
    combined.save("dcgan_combined_{}.hdf5".format(epoch))

history1000 = train(1000, 100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

save_models(1000)

history2000 = train(1000, 100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

save_models(2000)

history3000 = train(1000, 100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

save_models(3000)

history4000 = train(1000, 100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

save_models(4000)

history5000 = train(1000, 100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

save_models(5000)

history10000 = train(5000, 100)

show_images(0.5 * generator.predict(noise_data) + 0.5)

save_models(10000)


