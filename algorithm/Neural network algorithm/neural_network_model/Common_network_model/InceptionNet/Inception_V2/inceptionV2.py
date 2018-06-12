# -*- coding:utf-8 -*-

import tensorflow as tf
# from tensorflow.contrib import slim


x=tf.placeholder(tf.float32,[None,224,224,3],name='input_image')
num_classes=1000



def inception_block(net,filters=[],scope='block_1',activation_fn=tf.nn.relu):
    nets=[]
    with tf.variable_scope(scope):
        net1=tf.layers.conv2d(net,filters[0],1,1,'same',activation=activation_fn,name='conv1_1_1_s1')
        net1=tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1=tf.layers.conv2d(net,filters[1],1,1,'same',activation=activation_fn,name='conv2_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,filters[2],3,1,'same',activation=activation_fn,name='conv2_3_3_s1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1 = tf.layers.conv2d(net, filters[3], 1, 1, 'same', activation=activation_fn, name='conv3_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        # V1
        # net1 = tf.layers.conv2d(net1, filters[4], 5, 1, 'same', activation=activation_fn, name='conv3_5_5_s1')
        # V2
        net1 = tf.layers.conv2d(net1, filters[4],3, 1, 'same', activation=activation_fn, name='conv3_3_3_s1_0')
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.conv2d(net1, filters[4], 3, 1, 'same', activation=activation_fn, name='conv3_3_3_s1_1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1=tf.layers.max_pooling2d(net,3,1,'same',name='pool4_3_3_s1')
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,filters[5],1,1,'same',activation=activation_fn,name='conv4_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net=tf.concat(nets,-1) # [28,28,256]

    return net



def Inception_V2(x,softmax0=False,softmax1=False):
    net=x
    net = tf.layers.batch_normalization(net)
    net=tf.layers.conv2d(net,64,7,2,'same',activation=tf.nn.relu,name='conv1_7_7_s2') # [112,112,64]
    net = tf.layers.batch_normalization(net)
    net=tf.layers.max_pooling2d(net,3,2,'same',name='pool1_3_3_s2') # [56,56,64]
    # net=tf.nn.local_response_normalization(net)
    net = tf.layers.batch_normalization(net)

    net=tf.layers.conv2d(net,64,1,1,'valid',activation=tf.nn.relu,name='conv2_1_1_v1')# [56,56,64]
    net = tf.layers.batch_normalization(net)
    net = tf.layers.conv2d(net, 192, 3, 1, 'same', activation=tf.nn.relu, name='conv2_3_3_s1')  # [56,56,192]
    net = tf.layers.batch_normalization(net)
    # net = tf.nn.local_response_normalization(net)
    net = tf.layers.max_pooling2d(net, 3, 2, 'same', name='pool2_3_3_s2') # [28,28,192]
    net = tf.layers.batch_normalization(net)

    filters=[64,96,128,16,32,32]
    net=inception_block(net,filters,'block_3a') # [28,28,256]
    net = tf.layers.batch_normalization(net)
    filters = [128, 128, 192, 32, 96, 64]
    net=inception_block(net,filters,'block_3b') # [28,28,480]
    net = tf.layers.batch_normalization(net)

    net=tf.layers.max_pooling2d(net,3,2,'same',name='pool3_3_3_s2') # [14,14,480]
    net = tf.layers.batch_normalization(net)

    filters = [192, 96, 208, 16, 48, 64]
    net = inception_block(net, filters, 'block_4a')  # [14,14,512]
    net = tf.layers.batch_normalization(net)

    if softmax0:
        net1=tf.layers.average_pooling2d(net,14,1,'valid',name='softmax0_pool_14_14_v1') # [1,1,512]
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,1024,1,1,'same',activation=tf.nn.relu,name='softmax0_conv_1_1_s1') # [1,1,1024]
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.dropout(net1,0.4)
        net1=tf.layers.flatten(net1) # [-1,1*1*1024]
        net1 = tf.layers.dense(net1, 1024, activation=tf.nn.relu, name='softmax0_fc_1')  # [-1,2048]
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.dense(net1, num_classes, activation=tf.nn.softmax, name='softmax0_output_1')  # [-1,1000]
        return net1

    filters = [160, 112, 224, 24, 64, 64]
    net = inception_block(net, filters, 'block_4b')  # [14,14,512]
    net = tf.layers.batch_normalization(net)
    filters = [128, 128, 256, 24, 64, 64]
    net = inception_block(net, filters, 'block_4c')  # [14,14,512]
    net = tf.layers.batch_normalization(net)
    filters = [112, 144, 288, 32, 64, 64]
    net = inception_block(net, filters, 'block_4d')  # [14,14,528]
    net = tf.layers.batch_normalization(net)
    if softmax1:
        net1=tf.layers.average_pooling2d(net,14,1,'valid',name='softmax1_pool_14_14_v1') # [1,1,528]
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,1024,1,1,'same',activation=tf.nn.relu,name='softmax1_conv_1_1_s1') # [1,1,1024]
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.dropout(net1, 0.4)
        net1=tf.layers.flatten(net1) # [-1,1*1*1024]
        net1=tf.layers.dense(net1,1024,activation=tf.nn.relu,name='softmax1_fc_1') # [-1,2048]
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.dense(net1, num_classes, activation=tf.nn.softmax, name='softmax1_output_1') # [-1,1000]
        return net1

    filters = [256, 160, 320, 32, 128, 128]
    net = inception_block(net, filters, 'block_4e')  # [14,14,832]
    net = tf.layers.batch_normalization(net)
    net = tf.layers.max_pooling2d(net, 3, 2, 'same', name='pool4_3_3_s2')  # [7,7,832]
    net = tf.layers.batch_normalization(net)

    filters = [256, 160, 320, 32, 128, 128]
    net = inception_block(net, filters, 'block_5a')  # [7,7,832]
    net = tf.layers.batch_normalization(net)

    filters = [384, 192, 384, 48, 128, 128]
    net = inception_block(net, filters, 'block_5b')  # [7,7,1024]
    net = tf.layers.batch_normalization(net)

    net=tf.layers.average_pooling2d(net,7,1,'valid',name='pool5_7_7_v1') # [1,1,1024]
    net = tf.layers.batch_normalization(net)
    net=tf.layers.dropout(net,0.4)
    net=tf.layers.flatten(net)
    net=tf.layers.dense(net,num_classes,activation=None,name='logits')
    net=tf.nn.softmax(net,name='softmax')

    return net


pred=Inception_V2(x,True)

print(pred.shape)
