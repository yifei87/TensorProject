# -*- coding:utf-8 -*-
'''
resnet 50
resnet 101
resnet 152
'''
import tensorflow as tf


x=tf.placeholder(tf.float32,[None,224,224,3],name='input_image')
num_classes=1000

def resnet_block(net,scope='block_0',outputs=64,activation_fn=tf.nn.relu):
    with tf.variable_scope(scope):
        net1=net
        net1=tf.layers.conv2d(net1,outputs,3,1,'same',activation=None,name='conv1')
        net1 = tf.layers.batch_normalization(net1,name='BN1')
        net1=activation_fn(net1,name='relu1')

        net1 = tf.layers.conv2d(net1, outputs, 3, 1, 'same', activation=None, name='conv2')
        net1 = tf.layers.batch_normalization(net1, name='BN2')
        net1 = activation_fn(net1, name='relu2')


        if net.shape[-1]==net1.shape[-1]:
            net = tf.add(net, net1, name='add')
        else:
            # 由于net与net1维度不一致，可以对net的维度补零添加成与net1维度一致
            # 然后使用net=tf.add(net,net1,name='add') net要填充成net1维度一致
            # 等价与下面这种方式
            dmin=net.shape[-1]
            net=tf.add(net,net1[:,:,:,:dmin],name='add')
            net=tf.concat([net,net1[:,:,:,dmin:]],-1,name='concat')

        # inception 使用这种方式（合并通道）
        # net=tf.concat([net, net1], -1, name='concat')
        return net


def resnet_18(x):
    block_nmu = [2, 2, 2, 2] # （2+2+2+2）×2+2=18
    outputs = [64, 128, 256, 512]
    net=x
    net=tf.layers.batch_normalization(net,name='BN0')
    net=tf.layers.conv2d(net,64,(7,7),2,'SAME',activation=None,name='conv1') # [112,112,64]
    net = tf.layers.batch_normalization(net,name='BN1')
    net=tf.nn.relu(net,name='relu1')
    net=tf.layers.max_pooling2d(net,(3,3),2,'same',name='pool1') # [56,56,64]
    for index,b in enumerate(block_nmu):
        for i in range(b):
            net=resnet_block(net,'block%s_%s'%(index+1,i+1),outputs[index]) # [56,56,256]
        if index==len(block_nmu)-1:
            # 最后一个average pool
            net = tf.layers.average_pooling2d(net, 7, 2, 'valid', name='average_pool')
            # net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        else:
            net=tf.layers.max_pooling2d(net,2,2,'same',name='block%s_pool'%(index+1))

    # conv 代替fc
    net = tf.layers.conv2d(net, num_classes, 1,1,'SAME', activation=None,name='logits')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net=tf.nn.softmax(net, name='predictions')

    return net

pred=resnet_18(x)

def resnet_34(x):
    block_nmu = [3, 4, 6, 3] # （3+4+6+3）×2+2=34
    outputs = [64, 128, 256, 512]
    net=x
    net=tf.layers.batch_normalization(net,name='BN0')
    net=tf.layers.conv2d(net,64,(7,7),2,'SAME',activation=None,name='conv1') # [112,112,64]
    net = tf.layers.batch_normalization(net,name='BN1')
    net=tf.nn.relu(net,name='relu1')
    net=tf.layers.max_pooling2d(net,(3,3),2,'same',name='pool1') # [56,56,64]
    for index,b in enumerate(block_nmu):
        for i in range(b):
            net=resnet_block(net,'block%s_%s'%(index+1,i+1),outputs[index]) # [56,56,256]
        if index==len(block_nmu)-1:
            # 最后一个average pool
            net = tf.layers.average_pooling2d(net, 7, 2, 'valid', name='average_pool')
            # net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        else:
            net=tf.layers.max_pooling2d(net,2,2,'same',name='block%s_pool'%(index+1))

    # conv 代替fc
    net = tf.layers.conv2d(net, num_classes, 1,1,'SAME', activation=None,name='logits')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net=tf.nn.softmax(net, name='predictions')

    return net

