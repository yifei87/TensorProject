#! /usr/bin/python
# -*- coding: utf8 -*-

'''
Python：print显示颜色  https://www.cnblogs.com/ping-y/p/5897018.html
'''

import tensorflow as tf
import numpy as np
import prettytensor as pt


# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

n_output_layer = 10


# 定义待训练的神经网络
def convolutional_neural_network(data):
    net=tf.reshape(data, [-1, 28, 28, 1])
    net=tf.layers.conv2d(net,32,5,strides=1,padding='same',activation=tf.nn.relu)
    net=tf.layers.max_pooling2d(net,2,2,'same')
    net = tf.layers.dropout(net, 0.8)
    net = tf.layers.conv2d(net, 64, 5, strides=1, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2, 'same')
    net = tf.layers.dropout(net, 0.8)
    net=tf.layers.flatten(net)
    net = tf.layers.dense(net,1024,activation=tf.nn.relu)
    net=tf.layers.dropout(net,0.8)
    output=tf.layers.dense(net,n_output_layer,activation=None)

    return output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float', [None, 28 * 28*1])
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    # '''
    predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    '''
    # 使用prettytensor
    input_tensor = tf.reshape(X,[-1, 28, 28, 1])
    pretty_input = pt.wrap(input_tensor)
    predict, cost_func = (pretty_input. \
                          conv2d(kernel=5, depth=32, name='layer_conv1'). \
                          max_pool(kernel=2, stride=2). \
                          conv2d(kernel=5, depth=64, name='layer_conv2'). \
                          max_pool(kernel=2, stride=2). \
                          flatten(). \
                          fully_connected(size=1024, name='layer_fc1'). \
                          softmax_classifier(num_classes=n_output_layer, labels=Y)
                          )
    accuracy = predict.evaluate_classifier(Y)
    optimizer_ = tf.train.GradientDescentOptimizer(0.1)  # learning rate
    optimizer = pt.apply_optimizer(optimizer_, losses=[cost_func])
    '''

    epochs = 1
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for step in range(mnist.train.num_examples // batch_size):
                x, y = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                # epoch_loss += c
                # print(epoch, ' : ', epoch_loss)
                if step %20==0:
                    print('epoch',epoch,'\t|','step',step,'\t|','\033[1;35m train acc \033[0m',accuracy.eval({X:x,Y:y}),'\t|'
                          '\033[1;35m test acc \033[0m','\033[1;34m '+ str(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))+'\033[0m')

        print('准确率: ', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

train_neural_network(X, Y)