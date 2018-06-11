# -*- coding:utf-8 -*-
'''
可以参考：https://github.com/taki0112/Densenet-Tensorflow
densenet 121
densenet 169
densenet 201
densenet 161
'''
import tensorflow as tf


x=tf.placeholder(tf.float32,[None,224,224,3],name='input_image')
num_classes=1000

def densenet_block(net,scope='block_0',outputs=32,activation_fn=tf.nn.relu):
    with tf.variable_scope(scope):
        net1=net
        net1 = tf.layers.batch_normalization(net1, name='BN1')
        net1 = activation_fn(net1, name='relu1')
        net1=tf.layers.conv2d(net1,outputs*4,1,1,'same',activation=None,name='conv1')
        '''
        net = tf.add(net, net1, name='add1')
        dmin = net.shape[-1]
        net = tf.add(net, net1[:, :, :, :dmin], name='add')
        net = tf.concat([net, net1[:, :, :, dmin:]], -1, name='concat')
        net1 = net
        '''

        net1 = tf.layers.batch_normalization(net1, name='BN2')
        net1 = activation_fn(net1, name='relu2')
        net1 = tf.layers.conv2d(net1, outputs, 3, 1, 'same', activation=None, name='conv2')
        # net = tf.add(net, net1, name='add2')
        # net = tf.concat([net, net1], -1, name='concat')

        '''
        if net.shape[-1]==net1.shape[-1]:
            net = tf.add(net, net1, name='add')
        else:
            # 由于net与net1维度不一致，可以对net的维度补零添加成与net1维度一致
            # 然后使用net=tf.add(net,net1,name='add') net要填充成net1维度一致
            # 等价与下面这种方式
            dmin=net.shape[-1]
            net=tf.add(net,net1[:,:,:,:dmin],name='add')
            net=tf.concat([net,net1[:,:,:,dmin:]],-1,name='concat')
        '''
        # inception 使用这种方式（合并通道）
        # net=tf.concat([net, net1], -1, name='concat')
        return net1

# growth_rate=32
def densenet_121(x):
    block_nmu = [6, 12, 24, 16]
    growth_rate=32
    net=x
    net=tf.layers.batch_normalization(net,name='BN1')
    net = tf.nn.relu(net, name='relu1')
    net=tf.layers.conv2d(net,growth_rate*2,(7,7),2,'SAME',activation=None,name='conv1') # [112,112,32]
    net=tf.layers.max_pooling2d(net,(3,3),2,'same',name='pool1') # [56,56,32]
    for index,b in enumerate(block_nmu):
        nets=[]
        nets.append(net)
        for i in range(b):
            net=densenet_block(net,'block%s_%s'%(index+1,i+1),growth_rate) # [56,56,256]
            nets.append(net)
            net=tf.concat(nets,-1)

        net = tf.layers.batch_normalization(net, name='BN_tl_%s'%(index+1))
        net = tf.nn.relu(net, name='relu_tl_%s'%(index+1))
        net = tf.layers.conv2d(net, net.shape[-1]//2, (1, 1), 1, 'SAME', activation=None, name='conv_tl_%s'%(index+1))
        if index==len(block_nmu)-1:
            net = tf.layers.average_pooling2d(net, 7, 1, 'valid', name='ave_pool_tl_%s'%(index+1))
        else:
            net=tf.layers.average_pooling2d(net,2,2,'same',name='ave_pool_tl_%s'%(index+1))

    # net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    # conv 代替fc
    net = tf.layers.conv2d(net, num_classes, 1,1,'SAME', activation=None,name='logits')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net=tf.nn.softmax(net, name='predictions')

    return net

pred=densenet_121(x)

def densenet_169(x):
    block_nmu = [6, 12, 32, 32]
    growth_rate=32
    net=x
    net=tf.layers.batch_normalization(net,name='BN1')
    net = tf.nn.relu(net, name='relu1')
    net=tf.layers.conv2d(net,growth_rate*2,(7,7),2,'SAME',activation=None,name='conv1') # [112,112,32]
    net=tf.layers.max_pooling2d(net,(3,3),2,'same',name='pool1') # [56,56,32]
    for index,b in enumerate(block_nmu):
        nets=[]
        nets.append(net)
        for i in range(b):
            net=densenet_block(net,'block%s_%s'%(index+1,i+1),growth_rate) # [56,56,256]
            nets.append(net)
            net=tf.concat(nets,-1)

        net = tf.layers.batch_normalization(net, name='BN_tl_%s'%(index+1))
        net = tf.nn.relu(net, name='relu_tl_%s'%(index+1))
        net = tf.layers.conv2d(net, net.shape[-1]//2, (1, 1), 1, 'SAME', activation=None, name='conv_tl_%s'%(index+1))
        if index==len(block_nmu)-1:
            net = tf.layers.average_pooling2d(net, 7, 1, 'valid', name='ave_pool_tl_%s'%(index+1))
        else:
            net=tf.layers.average_pooling2d(net,2,2,'same',name='ave_pool_tl_%s'%(index+1))

    # net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    # conv 代替fc
    net = tf.layers.conv2d(net, num_classes, 1,1,'SAME', activation=None,name='logits')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net=tf.nn.softmax(net, name='predictions')

    return net

def densenet_201(x):
    block_nmu = [6, 12, 48, 32]
    growth_rate=32
    net=x
    net=tf.layers.batch_normalization(net,name='BN1')
    net = tf.nn.relu(net, name='relu1')
    net=tf.layers.conv2d(net,growth_rate*2,(7,7),2,'SAME',activation=None,name='conv1') # [112,112,32]
    net=tf.layers.max_pooling2d(net,(3,3),2,'same',name='pool1') # [56,56,32]
    for index,b in enumerate(block_nmu):
        nets=[]
        nets.append(net)
        for i in range(b):
            net=densenet_block(net,'block%s_%s'%(index+1,i+1),growth_rate) # [56,56,256]
            nets.append(net)
            net=tf.concat(nets,-1)

        net = tf.layers.batch_normalization(net, name='BN_tl_%s'%(index+1))
        net = tf.nn.relu(net, name='relu_tl_%s'%(index+1))
        net = tf.layers.conv2d(net, net.shape[-1]//2, (1, 1), 1, 'SAME', activation=None, name='conv_tl_%s'%(index+1))
        if index==len(block_nmu)-1:
            net = tf.layers.average_pooling2d(net, 7, 1, 'valid', name='ave_pool_tl_%s'%(index+1))
        else:
            net=tf.layers.average_pooling2d(net,2,2,'same',name='ave_pool_tl_%s'%(index+1))

    # net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    # conv 代替fc
    net = tf.layers.conv2d(net, num_classes, 1,1,'SAME', activation=None,name='logits')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net=tf.nn.softmax(net, name='predictions')

    return net

def densenet_161(x):
    block_nmu = [6, 12, 36, 24]
    growth_rate=48
    net=x
    net=tf.layers.batch_normalization(net,name='BN1')
    net = tf.nn.relu(net, name='relu1')
    net=tf.layers.conv2d(net,growth_rate*2,(7,7),2,'SAME',activation=None,name='conv1') # [112,112,32]
    net=tf.layers.max_pooling2d(net,(3,3),2,'same',name='pool1') # [56,56,32]
    for index,b in enumerate(block_nmu):
        nets=[]
        nets.append(net)
        for i in range(b):
            net=densenet_block(net,'block%s_%s'%(index+1,i+1),growth_rate) # [56,56,256]
            nets.append(net)
            net=tf.concat(nets,-1)

        net = tf.layers.batch_normalization(net, name='BN_tl_%s'%(index+1))
        net = tf.nn.relu(net, name='relu_tl_%s'%(index+1))
        net = tf.layers.conv2d(net, net.shape[-1]//2, (1, 1), 1, 'SAME', activation=None, name='conv_tl_%s'%(index+1))
        if index==len(block_nmu)-1:
            net = tf.layers.average_pooling2d(net, 7, 1, 'valid', name='ave_pool_tl_%s'%(index+1))
        else:
            net=tf.layers.average_pooling2d(net,2,2,'same',name='ave_pool_tl_%s'%(index+1))

    # net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    # conv 代替fc
    net = tf.layers.conv2d(net, num_classes, 1,1,'SAME', activation=None,name='logits')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    net=tf.nn.softmax(net, name='predictions')

    return net