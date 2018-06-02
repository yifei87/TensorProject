# -*- coding:utf-8 -*-
'''
更多梯度更新方法参考：https://blog.csdn.net/wc781708249/article/details/79290523

参考：
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb
'''
# import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1]) # [17,]
# train_X=numpy.expand_dims(train_X,-1) # [17,1]
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3]) # [17,]

# train_Y=numpy.expand_dims(train_Y,-1)

n_samples = train_X.shape[0]

# tf Graph Input
# X = tf.placeholder("float",[None,1])
# Y = tf.placeholder("float",[None,1])

# Set model weights
'''
with tf.variable_scope('D'):
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")
'''
# Construct a linear model
# pred = tf.add(tf.multiply(X, W), b) # y'=X*W+b
# pred=tf.layers.dense(X,1)

model = Sequential()
model.add(InputLayer(input_shape=(1,)))
model.add(Dense(1))
optimizer = Adam(lr=1e-3)
model.compile(optimizer,loss='MSE')

# Training
model.fit(x=train_X,
          y=train_Y,
          epochs=5, batch_size=1)

# Evaluation
result = model.evaluate(x=train_X,
                        y=train_Y)