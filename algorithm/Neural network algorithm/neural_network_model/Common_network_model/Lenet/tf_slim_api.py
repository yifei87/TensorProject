#! /usr/bin/python
# -*- coding:utf-8 -*-

'''
网络结构
[32,32,1]-->conv(f=5,s=2) [28,28,6]-->avgpool(f=2,s=2)[14,14,6]-->
conv(f=5,s=1)[10,10,6]-->avgpool(f=2,s=2)[5,5,6]-->fc(120)[None,120]
-->fc(84)[None,84]-->fc(10)[None,10]
'''

import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from tf_model_slim_lenets import lenet,lenet_arg_scope
slim = tf.contrib.slim

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

n_output_layer = 10

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
'''
X_train = mnist.train.images
X_train=np.reshape(X_train,[-1,28,28,1])
X_images=[]
for image in X_train:
    X_images.append(pad_image(image))
X_train=np.asarray(X_images)
y_train=mnist.train.labels
'''
X_test = mnist.test.images
X_test=np.reshape(X_test,[-1,28,28,1])
# X_images=[]
# for image in X_test:
#     X_images.append(pad_image(image))
# X_test=np.asarray(X_images)
y_test=mnist.test.labels

# 定义待训练的神经网络
def convolutional_neural_network(data):
    net=data # tf.reshape(data, [-1, 28, 28, 1])
    net=tf.layers.conv2d(net,6,5,strides=1,padding='valid',activation=tf.nn.relu)
    net=tf.layers.average_pooling2d(net,2,2,'same')
    net = tf.layers.dropout(net, 0.8)
    net = tf.layers.conv2d(net, 16, 5, strides=1, padding='valid', activation=tf.nn.relu)
    net = tf.layers.average_pooling2d(net, 2, 2, 'same')
    net = tf.layers.dropout(net, 0.8)
    net=tf.layers.flatten(net)
    net = tf.layers.dense(net,120,activation=tf.nn.relu)
    net=tf.layers.dropout(net,0.8)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.layers.dropout(net, 0.8)
    output=tf.layers.dense(net,n_output_layer,activation=tf.nn.softmax)

    return output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float', [None, 28,28,1])
Y = tf.placeholder('float')
is_training=tf.placeholder(tf.bool)

# 使用数据训练神经网络
def train_neural_network(X, Y):
    with slim.arg_scope(lenet_arg_scope(0.1)):
        predict,_=lenet(X,n_output_layer,is_training,0.8)
    # '''
    # predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=Y))
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
                x = np.reshape(x, [-1, 28, 28, 1])
                # X_images = []
                # for image in x:
                #     X_images.append(pad_image(image))
                # x = np.asarray(X_images)

                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y,is_training:True})
                # epoch_loss += c
                # print(epoch, ' : ', epoch_loss)
                if step %20==0:
                    print('epoch',epoch,'\t|','step',step,'\t|','\033[1;35m train acc \033[0m',accuracy.eval({X:x,Y:y,is_training:False}),'\t|'
                          '\033[1;35m test acc \033[0m','\033[1;34m '+ str(accuracy.eval({X: X_test, Y: mnist.test.labels,is_training:False}))+'\033[0m')

        print('准确率: ', accuracy.eval({X: X_test, Y: mnist.test.labels,is_training:False}))

train_neural_network(X, Y)