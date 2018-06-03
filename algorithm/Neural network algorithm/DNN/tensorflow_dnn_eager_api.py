# ####
# https://github.com/aymericdamien/TensorFlow-Examples
# https://github.com/nfmcclure/tensorflow_cookbook
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#####

from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
# Set Eager API
tfe.enable_eager_execution()


# Training Data
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

train_X = mnist.train.images # [n,784]
train_Y=mnist.train.labels # [n,]

n_samples = len(train_X)

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
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob =0.8 # tf.placeholder("float")
num_steps = 2000

# Using TF Dataset to split data into batches
dataset = tf.data.Dataset.from_tensor_slices(
    (train_X, train_Y)).batch(batch_size).shuffle(mnist.train.num_examples)
dataset_iter = tfe.Iterator(dataset)

# Define the neural network. To use eager API and tf.layers API together,
# we must instantiate a tfe.Network class as follow:
class LinearNet(tfe.Network):
    def __init__(self):
        # Define each layer
        super(LinearNet, self).__init__()
        # Hidden fully connected layer with 256 neurons
        # Output fully connected layer with a neuron for each class
        self.layer_1 = self.track_layer(tf.layers.Dense(n_hidden_1, activation=tf.nn.relu))
        self.layer_1_drop = self.track_layer(tf.layers.Dropout(dropout_keep_prob))

        self.layer_2 = self.track_layer(tf.layers.Dense(n_hidden_2, activation=tf.nn.relu))
        self.layer_2_drop = self.track_layer(tf.layers.Dropout(dropout_keep_prob))

        self.layer_3 = self.track_layer(tf.layers.Dense(n_hidden_3, activation=tf.nn.relu))
        self.layer_3_drop = self.track_layer(tf.layers.Dropout(dropout_keep_prob))

        self.layer_4 = self.track_layer(tf.layers.Dense(n_hidden_4, activation=tf.nn.relu))
        self.layer_4_drop = self.track_layer(tf.layers.Dropout(dropout_keep_prob))

        self.out_layer = self.track_layer(tf.layers.Dense(n_classes,activation=tf.nn.softmax))

    def call(self, x):
        x=self.layer_1(x)
        x=self.layer_1_drop(x)
        x=self.layer_2(x)
        x=self.layer_2_drop(x)
        x = self.layer_3(x)
        x = self.layer_3_drop(x)
        x = self.layer_4(x)
        x = self.layer_4_drop(x)
        return self.out_layer(x)

neural_net = LinearNet()

# Cross-Entropy loss function
def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy
                          (labels=labels,logits=inference_fn(inputs)))

# Calculate accuracy

def accuracy_fn(inference_fn, inputs, labels):
    prediction = inference_fn(inputs)
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# SGD Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Compute gradients
grad = tfe.implicit_gradients(loss_fn)

# Training
average_loss = 0.
average_acc = 0.
for step in range(num_steps):

    # Iterate through the dataset
    try:
        d = dataset_iter.next()
    except StopIteration:
        # Refill queue
        dataset_iter = tfe.Iterator(dataset)
        d = dataset_iter.next()

    # Images
    x_batch = d[0]
    # Labels
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # Compute the batch loss
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss
    # Compute the batch accuracy
    batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        # Display the initial cost, before optimizing
        print("Initial loss= {:.9f}".format(average_loss))

    # Update the variables following gradients info
    optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))

    # Display info
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

# Evaluate model on the test image set# Evalu
testX = mnist.test.images
testY = mnist.test.labels

test_acc = accuracy_fn(neural_net, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))