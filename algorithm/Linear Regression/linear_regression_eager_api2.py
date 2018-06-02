from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
# Set Eager API
tfe.enable_eager_execution()


# Training Data
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_X=np.expand_dims(train_X,-1) # [17,1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
train_Y=np.expand_dims(train_Y,-1) # [17,1]

n_samples = len(train_X)

# Parameters
learning_rate = 0.01
display_step = 100
num_steps = 1000

# Weight and Bias
# W = tfe.Variable(np.random.randn())
# b = tfe.Variable(np.random.randn())
batch_size=1

# Using TF Dataset to split data into batches
dataset = tf.data.Dataset.from_tensor_slices(
    (train_X, train_Y)).batch(batch_size)
dataset_iter = tfe.Iterator(dataset)


# Define the neural network. To use eager API and tf.layers API together,
# we must instantiate a tfe.Network class as follow:
class LinearNet(tfe.Network):
    def __init__(self):
        # Define each layer
        super(LinearNet, self).__init__()
        # Hidden fully connected layer with 256 neurons
        # Output fully connected layer with a neuron for each class
        self.out_layer = self.track_layer(tf.layers.Dense(1))

    def call(self, x):
        return self.out_layer(x)

neural_net = LinearNet()


# Cross-Entropy loss function
def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.losses.mean_squared_error(
        predictions =inference_fn(inputs), labels=labels))

# Calculate accuracy
'''
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''

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
    # x_batch=tf.expand_dims(x_batch,-1)
    # Labels
    y_batch = tf.cast(d[1], dtype=tf.int64)
    # y_batch = tf.expand_dims(y_batch, -1)
    # Compute the batch loss
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss
    # Compute the batch accuracy
    # batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    # average_acc += batch_accuracy

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
              "{:.9f}".format(average_loss))
        average_loss = 0.