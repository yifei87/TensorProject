# -*- coding:utf-8 -*-
'''
更多梯度更新方法参考：https://blog.csdn.net/wc781708249/article/details/79290523

参考：
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb
'''
import tensorflow as tf
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
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3]) # [17,]
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
with tf.variable_scope('D'):
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b) # y'=X*W+b

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# cost=tf.losses.mean_squared_error(labels=y,predictions=pred)

# Gradient descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# or
# tvars = tf.trainable_variables() # 获取所有可以更新的变量
# d_params = [v for v in tvars if v.name.startswith('D/')]
# optimizer =tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,var_list=d_params)

# or
optimizer=tf.train.AdamOptimizer(learning_rate)
gradients=optimizer.compute_gradients(cost,tf.trainable_variables()) # 省略tf.trainable_variables() 默认更新所有可更新的变量
# clipped_gradients = [(tf.clip_by_value(_[0], -1, 1), _[1]) for _ in gradients] # _[0] 对应dw ,_[1]对应db
# optimizer = optimizer.apply_gradients(clipped_gradients)
# or
optimizer=optimizer.apply_gradients(gradients)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
