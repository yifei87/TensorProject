# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define graph
x = tf.placeholder(tf.float32, [None, 28*28], name='x')
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

y = tf.nn.softmax(tf.matmul(x, W) + b, name='hypothesis')	# prediction value

y_ = tf.placeholder(tf.float32, [None, 10])	# true value
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Init
"""
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
"""

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sv=tf.train.Supervisor()
with sv.managed_session() as sess:
	# 注意：使用了sv = tf.train.Supervisor(),就不需要再初始化了,
	# 将sess.run(tf.initialize_all_variables())注释掉,否则会报错.

	# Train
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Write out 3 files
	# tf.train.Saver().save(sess, './model/trained.ckpt')
	sv.saver.save(sess, './model/trained.ckpt')
	tf.train.write_graph(sess.graph_def, '.', './model/trained.pb', as_text=False)
	tf.train.write_graph(sess.graph_def, '.', './model/trained.txt', as_text=True)

	print(sess.run(W))

	#'''
	# Eval

	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	#'''
