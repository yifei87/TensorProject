# -*- coding:utf-8 -*-

import tensorflow as tf
# from tensorflow.contrib import slim


x=tf.placeholder(tf.float32,[None,299,299,3],name='input_image')
num_classes=1000
keep_rate=tf.placeholder(tf.float32)
is_traning=tf.placeholder(tf.bool)

class Inception_V4(object):
    def __init__(self,x,num_classes,keep_rate,is_traning):
        self.x=x
        self.num_classes=num_classes
        self.activation_fn=tf.nn.relu
        self.keep_rate=keep_rate
        self.is_traning=is_traning

    def stem(self,net,scope='stem'):
        assert list(net.shape)[1:] == [299, 299, 3], 'stem: input shape wrong!'
        with tf.variable_scope(scope):
            net=tf.layers.batch_normalization(net,training=self.is_traning)
            net=tf.layers.conv2d(net,32,3,2,'valid',activation=self.activation_fn,name='conv1_3_3_v2') # [149,149,32]
            net = tf.layers.batch_normalization(net,training=self.is_traning)
            net=tf.layers.conv2d(net,32,3,1,'valid',activation=self.activation_fn,name='conv2_3_3_v1') # [147,147,32]
            net = tf.layers.batch_normalization(net,training=self.is_traning)
            net=tf.layers.conv2d(net,64,3,1,'same',activation=self.activation_fn,name='conv3_3_3_s1') # [147,147,64]
            net = tf.layers.batch_normalization(net,training=self.is_traning)
            net1=tf.layers.max_pooling2d(net,3,2,'valid',name='pool1_3_3_v2') # [73,73,64]
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net2=tf.layers.conv2d(net,96,3,2,'valid',activation=self.activation_fn,name='conv4_3_3_v2') # [73,73,96]
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            net=tf.concat([net1,net2],-1) # [73,73,160]
            net = tf.layers.batch_normalization(net,training=self.is_traning)

            net1=tf.layers.conv2d(net,64,1,1,'same',activation=self.activation_fn,name='conv5_1_1_s1') # [73,73,64]
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1=tf.layers.conv2d(net1,96,3,1,'valid',activation=self.activation_fn,name='conv6_3_3_v1') # [71,71,96]
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)

            net2=tf.layers.conv2d(net,64,1,1,'same',activation=self.activation_fn,name='conv7_1_1_s1')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            net2=tf.layers.conv2d(net2,64,(7,1),1,'same',activation=self.activation_fn,name='conv8_7_1_s1')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            net2 = tf.layers.conv2d(net2, 64, (1, 7), 1, 'same', activation=self.activation_fn, name='conv8_1_7_s1')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            net2 = tf.layers.conv2d(net2, 96, 3, 1, 'valid', activation=self.activation_fn, name='conv9_3_3_s1') # [71,71,96]
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            net = tf.concat([net1, net2], -1)  # [71,71,192]
            net = tf.layers.batch_normalization(net,training=self.is_traning)

            net1 = tf.layers.conv2d(net, 192, 3, 2, 'valid', activation=self.activation_fn,name='conv10_3_3_v2')  # [35,35,192]
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net2=tf.layers.max_pooling2d(net,3,2,'valid',name='pool2_3_3_v2')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)

            net=tf.concat([net1,net2],-1) # [35,35,384]
            net = tf.layers.batch_normalization(net,training=self.is_traning)

        assert list(net.shape)[1:]==[35,35,384] ,'stem: output shape wrong!'
        return net

    def inception_A(self,net,scope='inception_A_0'):
        assert list(net.shape)[1:] == [35, 35, 384], 'inception_A: input shape wrong!'
        nets=[]
        with tf.variable_scope(scope):
            net1=tf.layers.average_pooling2d(net,3,1,'same',name='avpool1_3_3_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1=tf.layers.conv2d(net1,96,1,1,'same',activation=self.activation_fn,name='conv1_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 96, 1, 1, 'same', activation=self.activation_fn, name='conv2_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1=tf.layers.conv2d(net,64,1,1,'same',activation=self.activation_fn,name='conv3_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1=tf.layers.conv2d(net1,96,3,1,'same',activation=self.activation_fn,name='conv3_3_3_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 64, 1, 1, 'same', activation=self.activation_fn, name='conv4_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 96, 3, 1, 'same', activation=self.activation_fn, name='conv4_3_3_s1_1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 96, 3, 1, 'same', activation=self.activation_fn, name='conv4_3_3_s1_2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net=tf.concat(nets,-1)
            net = tf.layers.batch_normalization(net,training=self.is_traning)

        assert list(net.shape)[1:] == [35, 35, 384], 'inception_A: input shape wrong!'

        return net

    def reduction_A(self,net,scope='reduction_A'):
        assert list(net.shape)[1:] == [35, 35, 384], 'reduction_A: input shape wrong!'
        nets=[]
        with tf.variable_scope(scope):
            net1=tf.layers.max_pooling2d(net,3,2,'valid',name='maxpool1_3_3_v2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1=tf.layers.conv2d(net,384,3,2,'valid',activation=self.activation_fn,name='conv2_3_3_v2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 192, 1, 1, 'same', activation=self.activation_fn, name='conv3_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 224, 3, 1, 'same', activation=self.activation_fn, name='conv3_3_3_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 256, 3, 2, 'valid', activation=self.activation_fn, name='conv3_3_3_v2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net=tf.concat(nets,-1)
            net = tf.layers.batch_normalization(net,training=self.is_traning)

        assert list(net.shape)[1:] == [17, 17, 1024], 'reduction_A: output shape wrong!'

        return net

    def inception_B(self, net, scope='inception_B_0'):
        assert list(net.shape)[1:] == [17, 17, 1024], 'inception_B: input shape wrong!'
        nets = []
        with tf.variable_scope(scope):
            net1 = tf.layers.average_pooling2d(net, 3, 1, 'same', name='avpool1_3_3_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 128, 1, 1, 'same', activation=self.activation_fn, name='conv1_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 384, 1, 1, 'same', activation=self.activation_fn, name='conv2_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 192, 1, 1, 'same', activation=self.activation_fn, name='conv3_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 224, (1,7), 1, 'same', activation=self.activation_fn, name='conv3_1_7_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 256, (7, 1), 1, 'same', activation=self.activation_fn, name='conv3_7_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 192, 1, 1, 'same', activation=self.activation_fn, name='conv4_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 192, (1,7), 1, 'same', activation=self.activation_fn, name='conv4_1_7_s1_1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 224, (7,1), 1, 'same', activation=self.activation_fn,name='conv4_7_1_s1_1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 224, (1,7), 1, 'same', activation=self.activation_fn,name='conv4_1_7_s1_2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 256, (7,1), 1, 'same', activation=self.activation_fn, name='conv4_7_1_s1_2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net = tf.concat(nets, -1)
            net = tf.layers.batch_normalization(net,training=self.is_traning)

        assert list(net.shape)[1:] == [17, 17, 1024], 'inception_B: input shape wrong!'

        return net

    def reduction_B(self,net,scope='reduction_B'):
        assert list(net.shape)[1:] == [17, 17, 1024], 'reduction_B: input shape wrong!'
        nets=[]
        with tf.variable_scope(scope):
            net1=tf.layers.max_pooling2d(net,3,2,'valid',name='maxpool1_3_3_v2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 192, 1, 1, 'same', activation=self.activation_fn, name='conv2_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1=tf.layers.conv2d(net1,192,3,2,'valid',activation=self.activation_fn,name='conv2_3_3_v2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 256, 1, 1, 'same', activation=self.activation_fn, name='conv3_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 256, (1,7), 1, 'same', activation=self.activation_fn, name='conv3_1_7_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 320, (7,1), 1, 'same', activation=self.activation_fn, name='conv3_7_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 320, 3, 2, 'valid', activation=self.activation_fn, name='conv3_3_3_v2')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net=tf.concat(nets,-1)
            net = tf.layers.batch_normalization(net,training=self.is_traning)

        assert list(net.shape)[1:] == [8, 8, 1536], 'reduction_B: output shape wrong!'

        return net

    def inception_C(self, net, scope='inception_C_0'):
        assert list(net.shape)[1:] == [8, 8, 1536], 'inception_C: input shape wrong!'
        nets = []
        with tf.variable_scope(scope):
            net1 = tf.layers.average_pooling2d(net, 3, 1, 'same', name='avpool1_3_3_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 256, 1, 1, 'same', activation=self.activation_fn, name='conv1_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 256, 1, 1, 'same', activation=self.activation_fn, name='conv2_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            nets.append(net1)

            net1 = tf.layers.conv2d(net, 384, 1, 1, 'same', activation=self.activation_fn, name='conv3_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net2 = tf.layers.conv2d(net1, 256, (1,3), 1, 'same', activation=self.activation_fn, name='conv3_1_3_s1')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            nets.append(net2)
            net2 = tf.layers.conv2d(net1, 256, (3, 1), 1, 'same', activation=self.activation_fn, name='conv3_3_1_s1')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            nets.append(net2)

            net1 = tf.layers.conv2d(net, 384, 1, 1, 'same', activation=self.activation_fn, name='conv4_1_1_s1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 448, (1,3), 1, 'same', activation=self.activation_fn, name='conv4_1_3_s1_1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net1 = tf.layers.conv2d(net1, 512, (3,1), 1, 'same', activation=self.activation_fn,name='conv4_3_1_s1_1')
            net1 = tf.layers.batch_normalization(net1,training=self.is_traning)
            net2 = tf.layers.conv2d(net1, 256, (1,3), 1, 'same', activation=self.activation_fn,name='conv4_1_3_s1_2')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            nets.append(net2)
            net2 = tf.layers.conv2d(net1, 256, (3,1), 1, 'same', activation=self.activation_fn, name='conv4_3_1_s1_2')
            net2 = tf.layers.batch_normalization(net2,training=self.is_traning)
            nets.append(net2)

            net = tf.concat(nets, -1)
            net = tf.layers.batch_normalization(net,training=self.is_traning)

        assert list(net.shape)[1:] == [8, 8, 1536], 'inception_C: input shape wrong!'

        return net

    def output(self,net,scope='output'):
        assert list(net.shape)[1:] == [8, 8, 1536], 'output: input shape wrong!'
        with tf.variable_scope(scope):
            net=tf.layers.average_pooling2d(net,8,1,'valid',name='avepool_8_8_v1') # [1,1,1536]
            net = tf.layers.batch_normalization(net,training=self.is_traning)

            net=tf.squeeze(net,[1,2]) # [-1,1536]

            net=tf.layers.dropout(net,self.keep_rate,training=self.is_traning)

            net=tf.layers.dense(net,self.num_classes,activation=tf.nn.softmax,name='output')

        assert list(net.shape)[-1] == self.num_classes, 'output: output shape wrong!'

        return net



inception=Inception_V4(x,num_classes,keep_rate,is_traning)
net=inception.x
net=inception.stem(net)
for i in range(4):
    net=inception.inception_A(net,'inception_A_%s'%(str(i)))
net=inception.reduction_A(net)

for i in range(7):
    net=inception.inception_B(net,'inception_B_%s'%(str(i)))
net=inception.reduction_B(net)

for i in range(3):
    net=inception.inception_C(net,'inception_C_%s'%(str(i)))

pred=inception.output(net)

print(pred.shape)