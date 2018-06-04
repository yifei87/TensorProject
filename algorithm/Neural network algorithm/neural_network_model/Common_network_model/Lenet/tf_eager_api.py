from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
# Set Eager API
tfe.enable_eager_execution() # 启动动态图（默认为静态图）
# 动态图中 tensor 转 numpy  a.numpy()  a is tensor
# numpy to tensor  tf.convert_to_tensor(a) a is numpy

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

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

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


X_train = mnist.train.images
X_train=np.reshape(X_train,[-1,28,28,1])
X_images=[]
for image in X_train:
    X_images.append(pad_image(image))
X_train=np.asarray(X_images)
y_train=mnist.train.labels

# Using TF Dataset to split data into batches
dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).batch(batch_size)
dataset_iter = tfe.Iterator(dataset)


# Define the neural network. To use eager API and tf.layers API together,
# we must instantiate a tfe.Network class as follow:
class NeuralNet(tfe.Network):
    def __init__(self):
        # Define each layer
        super(NeuralNet, self).__init__()
        # Hidden fully connected layer with 256 neurons

        self.conv1=self.track_layer(
            tf.layers.Conv2D(6, 5, strides=1, padding='valid', activation=tf.nn.relu)) # [n,28,28,32]
        self.pool1=self.track_layer(
            tf.layers.AveragePooling2D( 2, 2, 'same')) # [n,14,14,32]
        self.dropout1 = self.track_layer(
            tf.layers.Dropout(0.8))  # [n,14,14,32]

        self.conv2 = self.track_layer(
            tf.layers.Conv2D(16, 5, strides=1, padding='valid', activation=tf.nn.relu))  # [n,28,28,32]
        self.pool2 = self.track_layer(
            tf.layers.AveragePooling2D( 2, 2, 'same'))  # [n,14,14,32]
        self.dropout2 = self.track_layer(
            tf.layers.Dropout(0.8))  # [n,14,14,32]

        self.flatent=self.track_layer(
            tf.layers.Flatten())  # [n,14*14*32]

        self.fc1 = self.track_layer(
            tf.layers.Dense(120, activation=tf.nn.relu))
        # Hidden fully connected layer with 256 neurons
        self.fc2 = self.track_layer(
            tf.layers.Dense(84, activation=tf.nn.relu))
        # Output fully connected layer with a neuron for each class
        self.out_layer = self.track_layer(tf.layers.Dense(num_classes,activation=tf.nn.softmax))

    def call(self, x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x=self.flatent(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out_layer(x)


neural_net = NeuralNet()


# Cross-Entropy loss function
def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))

# Calculate accuracy
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
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
testX=np.reshape(testX,[-1,28,28,1])
X_images=[]
for image in testX:
    X_images.append(pad_image(image))
testX=np.asarray(X_images)
testY=mnist.test.labels

test_acc = accuracy_fn(neural_net, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))