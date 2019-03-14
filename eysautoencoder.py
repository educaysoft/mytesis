from IPython.display import Image, SVG
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
import gzip
img_size=512

def extract_data(filename, num_images):
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf.bytestream.read(img_size*img_size*num_images)
    data=np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
    data=data.reshape(num_images,img_size*img_size)
    return data
  
x_train_i=extract_data('data/input_train_images.idx3.gz',10)
x_train_o=extract_data('data/output_train_images.idx3.gz',10)

x_test_i=extract_data('data/input_test_images.idx3.gz',5)
x_test_o=extract_data('data/output_test_images.idx3.gz',5)

x_pred_i=extract_data('data/input_predict_images.idx3.gz',2)

max_value=float(x_train_i.max())
x_train_i=x_train_i.astype('float32')/max_value
x_train_o=x_train_o.astype('float32')/max_value
 
x_test_i=x_test_i.astype('float32')/max_value
x_test_o=x_test_o.astype('float32')/max_value

x_pred_i=x_pred_i.astype('float32')/max_value

x_train_i=x_train_i.reshape((len(x_train_i),img_size,img_size,1))
x_train_o=x_train_i.reshape((len(x_train_o),img_size,img_size,1))

x_test_i=x_test_i.reshape((len(x_test_i),img_size,img_size,1))
x_test_o=x_test_i.reshape((len(x_test_o),img_size,img_size,1))

x_pred_i=x_pred_i.reshape((len(x_pred_i),img_size,img_size,1))


autoencoder = Sequential()

autoencoder.add(Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(img_size,img_size,1)))
autoencoder.add(MaxPooling2D((2,2), padding='same'))
autoencoder.add(Conv2D(8,(3,3), activation= 'relu', padding='same'))
autoencoder.add(MaxPooling2D((2,2), padding='same'))
autoencoder.add(Conv2D(8,(3,3), strides=(2,2), activation='relu', padding='same'))
autoencoder.add(Flatten())
autoencoder.add(Reshape((64,64,8)))
autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(16, (3,3), activation='relu'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(1,(3,3),activation='sigmoid', padding='same'))

autoencoder.summary()

input_dim=x_train_i.shape[1]
input_img=Input(shape=(img_size,img_size,1))
encoder_layer = autoencoder.layers[0]
encoder=Model(input_img,encoder_layer(input_img))

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(x_train_i, x_train_o, epochs=20, batch_size=128, validation_data=(x_test_i,x_test_o))

'''
Obtiene la imagen libre de ruido
'''
encoded_imgs =encoder.predict(x_pred_i)
decoded_imgs=autoencoder.predict(x_pred_i)

plt.figure(1)
ax = plt.subplot(1,1, 1)
plt.imshow(decoded_imgs[0].reshape(img_size,img_size))
plt.gray()
plt.show()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


