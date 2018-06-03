# -*- coding:utf-8 -*-
"""
 Deeper Multi-Layer Pecptron with XAVIER Init
 Xavier init from {Project: https://github.com/aymericdamien/TensorFlow-Examples/}
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Xavier Init
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

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
net=tf.layers.dense(x,n_hidden_1,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

net=tf.layers.dense(net,n_hidden_2,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

net=tf.layers.dense(net,n_hidden_3,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

net=tf.layers.dense(net,n_hidden_4,activation=tf.nn.relu)
net=tf.layers.dropout(net,dropout_keep_prob)

pred=tf.layers.dense(net,n_classes,activation=tf.nn.softmax)
# Construct model
# pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost) # Adam Optimizer

# Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

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

