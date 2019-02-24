from IPython.display import Image, SVG
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
(x_train,_),(x_test,_) = mnist.load_data()
max_value=float(x_train.max())
x_train = x_train.astype('float32')/max_value
x_test = x_test.astype('float32')/max_value

x_train = x_train.reshape((len(x_train), 28,28,1))
x_test = x_test.reshape((len(x_test), 28,28,1))

autoencoder = Sequential()

autoencoder.add(Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
autoencoder.add(MaxPooling2D((2,2), padding='same'))
autoencoder.add(Conv2D(8,(3,3), activation= 'relu', padding='same'))
autoencoder.add(MaxPooling2D((2,2), padding='same'))
autoencoder.add(Conv2D(8,(3,3), strides=(2,2), activation='relu', padding='same'))
autoencoder.add(Flatten())
autoencoder.add(Reshape((4,4,8)))
autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(16, (3,3), activation='relu'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(1,(3,3),activation='sigmoid', padding='same'))

input_dim=x_train.shape[1]
input_img=Input(shape=(28,28,1))
encoder_layer = autoencoder.layers[0]
encoder=Model(input_img,encoder_layer(input_img))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train,xtrain,epochs=20, batch_size=128, validation_data=(x_test, x_test))
num_images=10
np.random.seed(42)
random_test_images= np.random.randint(x_test.shape[0],size=num_images)

encoded_imgs =encoder.predict(x_test)
decoded_imgs=autoencoder.predict(x_test)

plt.figure(figsize=(18,4))
for i, image_idx in enumerate(random_test_images):
  ax = plt.subplot(3, num_images, i+1)
  plt.imshow(x_test[image_idx].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  
  ax =plt.subplot(3,num_images, 2*num_images+i+1)
  plt.imshow(decoded_imgs[image_idx].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)
  plt.show()

