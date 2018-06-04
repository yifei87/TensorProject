#! /usr/bin/python
# -*- coding:utf-8 -*-

'''
网络结构
[32,32,1]-->conv(f=5,s=2) [28,28,6]-->avgpool(f=2,s=2)[14,14,6]-->
conv(f=5,s=1)[10,10,6]-->avgpool(f=2,s=2)[5,5,6]-->fc(120)[None,120]
-->fc(84)[None,84]-->fc(10)[None,10]
'''

import numpy as np
np.random.seed(1337)  # for reproducibility
# from keras.datasets import mnist
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,AveragePooling2D
from keras.optimizers import Adam

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

def pad_image(image,pad_h=32,pad_w=32):
    h, w = image.shape[:2]
    assert h<pad_h
    assert w<pad_w

    err_h=pad_h-h
    top_pad=err_h//2
    bottom_pad=err_h-top_pad

    err_w = pad_w - w
    left_pad = err_w // 2
    right_pad = err_w - left_pad

    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image

X_train = mnist.train.images
X_train=np.reshape(X_train,[-1,28,28,1])
X_images=[]
for image in X_train:
    X_images.append(pad_image(image))
X_train=np.asarray(X_images)
y_train=mnist.train.labels

X_test = mnist.test.images
X_test=np.reshape(X_test,[-1,28,28,1])
X_images=[]
for image in X_test:
    X_images.append(pad_image(image))
X_test=np.asarray(X_images)
y_test=mnist.test.labels

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (None, 28, 28,6)
model.add(Convolution2D(
    batch_input_shape=(None, 32, 32,1), # theano  batch_input_shape=(None, 1, 28, 28),
    filters=6,
    kernel_size=5,
    strides=1,
    padding='valid',     # Padding method
    data_format= 'channels_last',#'channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (None, 14, 14,6)
model.add(AveragePooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_last',
))

# Conv layer 2 output shape (None, 10, 10,16)
model.add(Convolution2D(16, 5, strides=1, padding='valid', data_format='channels_last'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape ( None，5, 5，16)
model.add(AveragePooling2D(2, 2, 'same', data_format='channels_last'))

# Fully connected layer 1 input shape (5 * 5 * 16) = (400), output shape (120)
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dense(84))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
