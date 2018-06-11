# -*- coding:utf-8 -*-

'''
使用fashion-mnist数据（类似于mnist）下载保存到 MNIST_data文件夹
https://github.com/zalandoresearch/fashion-mnist
'''

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from alexnet import alexnet_v2_arg_scope,alexnet_v2

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

n_output_layer = 10
img_h=224
img_w=224
batch_size=64

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
X_images=[]
for image in X_test:
    X_images.append(pad_image(image,img_h,img_w))
X_test=np.asarray(X_images)
y_test=mnist.test.labels

# -------------------------------#

X = tf.placeholder('float', [None, img_h,img_w,1])
Y = tf.placeholder('float')
is_training=tf.placeholder(tf.bool)


with slim.arg_scope(alexnet_v2_arg_scope()):
    predict, _ = alexnet_v2(X, n_output_layer, is_training, 0.8)

cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=Y))
optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001
correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


epochs = 1
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    epoch_loss = 0
    for epoch in range(epochs):
        for step in range(mnist.train.num_examples // batch_size):
            x, y = mnist.train.next_batch(batch_size)
            x = np.reshape(x, [-1, 28, 28, 1])
            X_images = []
            for image in x:
                X_images.append(pad_image(image,img_h,img_w))
            x = np.asarray(X_images)

            _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y,is_training:True})
            # epoch_loss += c
            # print(epoch, ' : ', epoch_loss)
            if step %20==0:
                print('epoch',epoch,'\t|','step',step,'\t|','\033[1;35m train acc \033[0m',accuracy.eval({X:x,Y:y,is_training:False}),'\t|'
                      '\033[1;35m test acc \033[0m','\033[1;34m '+ str(accuracy.eval({X: X_test[:batch_size], Y: mnist.test.labels[:batch_size],is_training:False}))+'\033[0m')

    acc=[]
    for step in range(mnist.test.num_examples // batch_size):
        x, y = mnist.test.next_batch(batch_size)
        x = np.reshape(x, [-1, 28, 28, 1])
        X_images = []
        for image in x:
            X_images.append(pad_image(image, img_h, img_w))
        x = np.asarray(X_images)
        acc1=accuracy.eval({X: x, Y: y,is_training:False})
        acc.append(acc1)
        print('准确率: ', acc1)

    print('平均准确率：',np.mean(acc))