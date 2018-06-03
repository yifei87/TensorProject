# -*- coding:utf-8 -*-
"""
 pip install prettytensor
 windows运行会报错 ？？？
 Deeper Multi-Layer Pecptron with XAVIER Init
 Xavier init from {Project: https://github.com/aymericdamien/TensorFlow-Examples/}
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import prettytensor as pt
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Parameters
learning_rate   = 0.001
training_epochs = 50
batch_size      = 100
display_step    = 1

# Network Parameters
n_input    = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_hidden_3 = 256 # 3rd layer num features
n_hidden_4 = 256 # 4th layer num features
n_classes  = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder("float")

# Create model
'''
# using layer api
net=tf.layers.dense(x,n_hidden_1,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

net=tf.layers.dense(net,n_hidden_2,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

net=tf.layers.dense(net,n_hidden_3,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

net=tf.layers.dense(net,n_hidden_4,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

pred=tf.layers.dense(net,n_classes,activation=tf.nn.softmax)
'''
# using slim api
'''
net=x
net=slim.fully_connected(net,n_hidden_1,activation_fn=tf.nn.relu)
net=slim.dropout(net,dropout_keep_prob)
net=slim.fully_connected(net,n_hidden_2,activation_fn=tf.nn.relu)
net=slim.dropout(net,dropout_keep_prob)
net=slim.fully_connected(net,n_hidden_3,activation_fn=tf.nn.relu)
net=slim.dropout(net,dropout_keep_prob)
net=slim.fully_connected(net,n_hidden_4,activation_fn=tf.nn.relu)
net=slim.dropout(net,dropout_keep_prob)
pred=slim.fully_connected(net,n_classes,activation_fn=tf.nn.softmax)
'''
# using pretty tensor
input_tensor=x
pretty_input = pt.wrap(input_tensor)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    pred,cost=(pretty_input. \
               fully_connected(size=n_hidden_1,name='fc1').dropout(dropout_keep_prob). \
               fully_connected(size=n_hidden_2,name='fc2').dropout(dropout_keep_prob). \
               fully_connected(size=n_hidden_3,name='fc3').dropout(dropout_keep_prob). \
               fully_connected(size=n_hidden_4,name='fc4').dropout(dropout_keep_prob). \
               softmax_classifier(num_classes=n_classes, labels=y))

accuracy = pred.evaluate_classifier(y)
optimizer_ = tf.train.GradientDescentOptimizer(learning_rate)  # learning rate
optimizer = pt.apply_optimizer(optimizer_, losses=[cost])

# Initializing the variables
init = tf.global_variables_initializer()

print ("Network Ready")

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})/total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})
        print ("Training accuracy: %.3f" % (train_acc))

print ("Optimization Finished!")

test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, dropout_keep_prob:1.})
print ("Training accuracy: %.3f" % (test_acc))

