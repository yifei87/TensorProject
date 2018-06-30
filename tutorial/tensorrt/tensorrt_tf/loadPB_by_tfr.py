# -*- coding:utf-8 -*-
'''
使用tensorflow自带的tensorrt （tensorflow GPU 1.7以上）
frozen_graph-->trt_graph

使用tensorrt方式加载冻结的pb文件（tf的冻结图转成tfr模式来加速）
'''

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
# or
# import trt_convert as trt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

frozen_graph='./model/frozen_graph.pb'

output_names=['hypothesis']
input_names=['x']

# tensorrt加载冻结网络
def get_trt_graph(frozen_graph,outputs,max_batch_size=1,
                  max_workspace_size_bytes=1<<25,mode='FP16',
                  minimum_segment_size=2):
    """Return trt converted graph."""
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        input_graph_def = tf.GraphDef()
        input_graph_def.ParseFromString(f.read())

    if mode in ["FP32", "FP16", "INT8"]:
        return trt.create_inference_graph(
            input_graph_def=input_graph_def,
            outputs=outputs, # list
            max_batch_size=max_batch_size,
            max_workspace_size_bytes=max_workspace_size_bytes,
            precision_mode=mode,
            minimum_segment_size=minimum_segment_size
        )
    return None

# 纯tensorflow加载冻结网络
def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    """
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",# prefix
            op_dict=None,
            producer_op_list=None
        )
    """
    return graph_def # graph


# tensorrt模式
trt_graph = get_trt_graph(
    frozen_graph=frozen_graph,
    outputs=output_names,
    max_batch_size=10000,)

# tensorflow模式
# trt_graph=load_graph(frozen_graph)



tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_output = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')
y_ = tf.placeholder(tf.float32, [None, 10])  # true value

correct_prediction = tf.equal(tf.argmax(tf_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(tf_sess.run(accuracy, feed_dict={tf_input: mnist.test.images, y_: mnist.test.labels}))

tf_sess.close()