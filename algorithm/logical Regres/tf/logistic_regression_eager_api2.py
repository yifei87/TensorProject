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
learning_rate = 0.1
training_epochs = 25
batch_size = 100
display_step = 1
num_steps = 2000
# x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
# y = tf.placeholder(tf.int64, [None,])

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
        self.out_layer = self.track_layer(tf.layers.Dense(10,activation=tf.nn.softmax))

    def call(self, x):
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