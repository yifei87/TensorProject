# Our simple RNN consists of
- One input layer which converts a $28$ dimensional input to an $128$ dimensional hidden layer,
- One intermediate recurrent neural network (LSTM)
- One output layer which converts an $128$ dimensional output of the LSTM to $10$ dimensional output indicating a class label.

![这里写图片描述](https://img-blog.csdn.net/20180602093953853?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# Out Network looks like this
![这里写图片描述](https://img-blog.csdn.net/20180602094019262?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##rnn.static_rnn
[5,28,28]-->[28*5,28]-->[28*5,128]-->28个 [5,128]--->取最后一个状态[5,128]-->[5,10]

##dynamic_rnn
[5,28,28]-->[5,28,128]-->28个 [5,128]--->取最后一个状态[5,128]-->[5,10]

参考：http://note.youdao.com/noteshare?id=e30289a9b5e8e1b69f19a7de4e794abf

完整参考：
```
# Sequence classification with LSTM

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
# %matplotlib inline
print ("Packages imported")

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimgs, trainlabels, testimgs, testlabels \
 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
ntrain, ntest, dim, nclasses \
 = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print ("MNIST loaded")

# Contruct a Recurrent Neural Network
diminput  = 28
dimhidden = 128
dimoutput = nclasses
nsteps    = 28
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}

def _RNN(_X, _istate, _W, _b, _nsteps, _name):
    # 1. Permute input from [batchsize, nsteps, diminput]
    #   => [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput]
    _X = tf.reshape(_X, [-1, diminput])
    # 3. Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 4. Splite data to 'nsteps' chunks. An i-th chunck indicates i-th batch data
    _Hsplit = tf.split(_H, _nsteps,0)
    # 5. Get LSTM's final output (_LSTM_O) and state (_LSTM_S)
    #    Both _LSTM_O and _LSTM_S consist of 'batchsize' elements
    #    Only _LSTM_O will be used to predict the output.
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.contrib.rnn.static_rnn(lstm_cell, _Hsplit,dtype=tf.float32)#, initial_state=_istate)
    # 6. Output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']
    # Return!
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }
print ("Network ready")

# Out Network looks like this

# Define functions
learning_rate = 0.001
x      = tf.placeholder("float", [None, nsteps, diminput])
istate = tf.placeholder("float", [None, 2*dimhidden])
    # state & cell => 2x n_hidden
y      = tf.placeholder("float", [None, dimoutput])
myrnn  = _RNN(x, istate, weights, biases, nsteps, 'basic')
pred   = myrnn['O']
cost   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optm   = tf.train.AdamOptimizer(learning_rate).minimize(cost) # Adam Optimizer
accr   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))
init   = tf.global_variables_initializer()
print ("Network Ready!")

# run
training_epochs = 5
batch_size      = 128
display_step    = 1
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter('/tmp/tensorflow_logs', graph=sess.graph)
print ("Start optimization")
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
        # Fit training using batch data
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
        sess.run(optm, feed_dict=feeds)
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict=feeds)/total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
        train_acc = sess.run(accr, feed_dict=feeds)
        print (" Training accuracy: %.3f" % (train_acc))
        testimgs = testimgs.reshape((ntest, nsteps, diminput))
        feeds = {x: testimgs, y: testlabels, istate: np.zeros((ntest, 2*dimhidden))}
        test_acc = sess.run(accr, feed_dict=feeds)
        print (" Test accuracy: %.3f" % (test_acc))
print ("Optimization Finished.")
```