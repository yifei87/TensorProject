#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
# from keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Activation, Dense,LSTM
from tensorflow.python.keras.optimizers import Adam

TIME_STEPS = 28     # same as the height of the image
INPUT_SIZE = 28     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 128
LR = 0.001

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# data pre-processing
X_train =mnist.train.images
X_test = mnist.test.images
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
y_train = mnist.train.labels
y_test = mnist.test.labels

# build RNN model
model = Sequential()

# RNN cell
model.add(LSTM(CELL_SIZE, batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
               dropout=0.2))  # LSTM 层

model.add(Dense(OUTPUT_SIZE))    # 二分类层

model.add(Activation('softmax')) # Sigmoid

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)