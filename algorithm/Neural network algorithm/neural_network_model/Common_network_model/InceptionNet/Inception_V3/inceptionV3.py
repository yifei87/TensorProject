# -*- coding:utf-8 -*-

import tensorflow as tf
# from tensorflow.contrib import slim


x=tf.placeholder(tf.float32,[None,299,299,3],name='input_image')
num_classes=1000



def inception_block_1(net,filters=[],scope='block_1',activation_fn=tf.nn.relu):
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

def inception_block_2(net,filters=[],scope='block_1',activation_fn=tf.nn.relu):
    nets=[]
    with tf.variable_scope(scope):
        net1=tf.layers.conv2d(net,filters[0],1,1,'same',activation=activation_fn,name='conv1_1_1_s1')
        net1=tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1=tf.layers.conv2d(net,filters[1],1,1,'same',activation=activation_fn,name='conv2_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        # net1=tf.layers.conv2d(net1,filters[2],3,1,'same',activation=activation_fn,name='conv2_3_3_s1')
        net1 = tf.layers.conv2d(net1, filters[2], (1,7), 1, 'same', activation=activation_fn, name='conv2_1_7_s1')
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.conv2d(net1, filters[2], (7, 1), 1, 'same', activation=activation_fn, name='conv2_7_1_s1')

        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1 = tf.layers.conv2d(net, filters[3], 1, 1, 'same', activation=activation_fn, name='conv3_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        # V1
        # net1 = tf.layers.conv2d(net1, filters[4], 5, 1, 'same', activation=activation_fn, name='conv3_5_5_s1')
        # V2
        # net1 = tf.layers.conv2d(net1, filters[4],3, 1, 'same', activation=activation_fn, name='conv3_3_3_s1_0')
        net1 = tf.layers.conv2d(net1, filters[4], (1,7), 1, 'same', activation=activation_fn, name='conv3_1_7_s1_0')
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.conv2d(net1, filters[4], (7, 1), 1, 'same', activation=activation_fn, name='conv3_7_1_s1_0')

        net1 = tf.layers.batch_normalization(net1)
        # net1 = tf.layers.conv2d(net1, filters[4], 3, 1, 'same', activation=activation_fn, name='conv3_3_3_s1_1')
        net1 = tf.layers.conv2d(net1, filters[4], (1, 7), 1, 'same', activation=activation_fn, name='conv3_1_7_s1_1')
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.conv2d(net1, filters[4], (7, 1), 1, 'same', activation=activation_fn, name='conv3_7_1_s1_1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1=tf.layers.max_pooling2d(net,3,1,'same',name='pool4_3_3_s1')
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,filters[5],1,1,'same',activation=activation_fn,name='conv4_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net=tf.concat(nets,-1) # [28,28,256]

    return net

def inception_block_3(net,filters=[],scope='block_1',activation_fn=tf.nn.relu):
    nets=[]
    with tf.variable_scope(scope):
        net1=tf.layers.conv2d(net,filters[0],1,1,'same',activation=activation_fn,name='conv1_1_1_s1')
        net1=tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1=tf.layers.conv2d(net,filters[1],1,1,'same',activation=activation_fn,name='conv2_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,filters[2],(1,3),1,'same',activation=activation_fn,name='conv2_1_3_s1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)
        net1 = tf.layers.conv2d(net1, filters[2], (3,1), 1, 'same', activation=activation_fn, name='conv2_3_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1 = tf.layers.conv2d(net, filters[3], 1, 1, 'same', activation=activation_fn, name='conv3_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        # V1
        # net1 = tf.layers.conv2d(net1, filters[4], 5, 1, 'same', activation=activation_fn, name='conv3_5_5_s1')
        # V2
        net1 = tf.layers.conv2d(net1, filters[4],3, 1, 'same', activation=activation_fn, name='conv3_3_3_s1_0')
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.conv2d(net1, filters[4], (1,3), 1, 'same', activation=activation_fn, name='conv3_1_3_s1_1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)
        net1 = tf.layers.conv2d(net1, filters[4], (3,1), 1, 'same', activation=activation_fn, name='conv3_3_1_s1_1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net1=tf.layers.max_pooling2d(net,3,1,'same',name='pool4_3_3_s1')
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,filters[5],1,1,'same',activation=activation_fn,name='conv4_1_1_s1')
        net1 = tf.layers.batch_normalization(net1)
        nets.append(net1)

        net=tf.concat(nets,-1) # [28,28,256]

    return net



def Inception_V3(x,softmax0=False,softmax1=False):
    net=x
    net = tf.layers.batch_normalization(net)
    net=tf.layers.conv2d(net,32,3,2,'valid',activation=tf.nn.relu,name='conv1_3_3_v2_0') # [149,149,32]
    net = tf.layers.batch_normalization(net)
    net = tf.layers.conv2d(net, 32, 3, 1, 'valid', activation=tf.nn.relu, name='conv1_3_3_v2_1')  # [147,147,32]
    net = tf.layers.batch_normalization(net)
    net = tf.layers.conv2d(net, 64, 3, 1, 'same', activation=tf.nn.relu, name='conv1_3_3_s2_2')  # [147,147,64]
    net = tf.layers.batch_normalization(net)
    net=tf.layers.max_pooling2d(net,3,2,'valid',name='pool1_3_3_v2') # [73,73,64]
    # net=tf.nn.local_response_normalization(net)
    net = tf.layers.batch_normalization(net)

    net=tf.layers.conv2d(net,80,3,1,'valid',activation=tf.nn.relu,name='conv2_3_3_v1_0') # [71,71,80]
    net = tf.layers.batch_normalization(net)
    net = tf.layers.conv2d(net, 192, 3, 2, 'valid', activation=tf.nn.relu, name='conv2_3_3_v2_1')  # [35,35,192]
    net = tf.layers.batch_normalization(net)
    net = tf.layers.conv2d(net, 288, 3, 1, 'same', activation=tf.nn.relu, name='conv2_3_3_s1_2')  # [35,35,288]
    net = tf.layers.batch_normalization(net)

    filters = [128, 128, 192, 32, 96, 64]
    net=inception_block_1(net,filters,'block_3a') # [35,35,480]
    net = tf.layers.batch_normalization(net)
    filters = [160, 128, 224, 64, 128, 96]
    net = inception_block_1(net, filters, 'block_3b')  # [35,35,608]
    net = tf.layers.batch_normalization(net)
    filters = [192, 128, 256, 96, 160, 160]
    net = inception_block_1(net, filters, 'block_3c')  # [35,35,768]
    net = tf.layers.batch_normalization(net)

    net=tf.layers.max_pooling2d(net,3,2,'valid',name='pool3_3_3_v2') # [17,17,768]
    net = tf.layers.batch_normalization(net)


    filters = [224, 128, 288, 96, 192, 160]
    net = inception_block_2(net, filters, 'block_4a')  # [17,17,864]
    net = tf.layers.batch_normalization(net)
    if softmax0:
        net1=tf.layers.average_pooling2d(net,17,1,'valid',name='softmax0_pool_17_17_v3') # [1,1,864]
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,1024,1,1,'same',activation=tf.nn.relu,name='softmax0_conv_1_1_s1') # [4,4,1024]
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.dropout(net1,0.4)
        net1=tf.layers.flatten(net1) # [-1,4*4*1024]
        net1 = tf.layers.dense(net1, 1024, activation=tf.nn.relu, name='softmax0_fc_1')  # [-1,2048]
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.dense(net1, num_classes, activation=tf.nn.softmax, name='softmax0_output_1')  # [-1,1000]
        return net1
    filters = [256, 160, 320, 128, 192, 192]
    net = inception_block_2(net, filters, 'block_4b')  # [17,17,960]
    net = tf.layers.batch_normalization(net)
    filters = [288, 160, 352, 128, 224, 224]
    net = inception_block_2(net, filters, 'block_4c')  # [17,17,1088]
    net = tf.layers.batch_normalization(net)
    filters = [320, 192, 384, 128, 256, 256]
    net = inception_block_2(net, filters, 'block_4d')  # [17,17,1216]
    net = tf.layers.batch_normalization(net)

    if softmax1:
        net1=tf.layers.average_pooling2d(net,17,1,'valid',name='softmax1_pool_17_17_v3') # [1,1,1216]
        net1 = tf.layers.batch_normalization(net1)
        net1=tf.layers.conv2d(net1,1024,1,1,'same',activation=tf.nn.relu,name='softmax1_conv_1_1_s1') # [4,4,1024]
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.dropout(net1, 0.4)
        net1=tf.layers.flatten(net1) # [-1,4*4*1024]
        net1=tf.layers.dense(net1,1024,activation=tf.nn.relu,name='softmax1_fc_1') # [-1,2048]
        net1 = tf.layers.batch_normalization(net1)
        net1 = tf.layers.dense(net1, num_classes, activation=tf.nn.softmax, name='softmax1_output_1') # [-1,1000]
        return net1
    filters = [352, 208, 416, 128, 256, 256]
    net = inception_block_2(net, filters, 'block_4e')  # [17,17,1280]
    net = tf.layers.batch_normalization(net)

    net = tf.layers.max_pooling2d(net, 3, 2, 'valid', name='pool4_3_3_v2')  # [8,8,1280]
    net = tf.layers.batch_normalization(net)

    filters = [352, 208, 416, 128, 256, 256]
    net = inception_block_3(net, filters, 'block_5a')  # [8,8,1952]
    net = tf.layers.batch_normalization(net)

    filters = [352, 208, 448, 128, 256, 288]
    net = inception_block_3(net, filters, 'block_5b')  # [8,8,2048]
    net = tf.layers.batch_normalization(net)

    net=tf.layers.average_pooling2d(net,8,1,'valid',name='ave_pool_8_8_v1') # [1,1,2048]

    net = tf.layers.batch_normalization(net)
    net=tf.layers.dropout(net,0.4)
    net=tf.layers.flatten(net)
    net=tf.layers.dense(net,num_classes,activation=None,name='logits')
    net=tf.nn.softmax(net,name='softmax')

    return net


pred=Inception_V3(x)

print(pred.shape)
