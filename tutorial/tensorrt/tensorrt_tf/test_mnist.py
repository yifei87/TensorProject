from PIL import Image
from numpy import *
import tensorflow as tf
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
im = image.open("./")
img = array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
data = img.reshape([1, 784])

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
'''

graph_path = './model/frozen_graph.pb'
with open(graph_path, 'rb') as f:
	proto_b = f.read()
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(proto_b)
	#tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='')
	x, W, b, hypothesis = tf.import_graph_def(graph_def, return_elements=['x:0', 'W:0', 'b:0', 'hypothesis'], name='')
	print(W)

#print(graph_def)

with tf.Session() as sess:
	### saver.restore : load checkpoint(=weights)	###
	'''
	sess.run(init_op)
	save_path = "/home/titanxp/prac_tensor/tensorrt_test/model/trained.ckpt"
	saver.restore(sess, save_path)
	'''

	#x = sess.graph.get_tensor_by_name("x:0")
	#W = sess.graph.get_tensor_by_name("W:0")
	#b = sess.graph.get_tensor_by_name("b:0")
	#hypothesis = sess.graph.get_operation_by_name("hypothesis")

	print(hypothesis)
	print(sess.run(W))

	predictions = sess.run(hypothesis, feed_dict={x: mnist.test.images})
	print(predictions)
