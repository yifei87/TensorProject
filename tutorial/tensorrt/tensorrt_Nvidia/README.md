参考：

- [tensorrt API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/index.html)
- [facenet.py](https://github.com/davidsandberg/facenet/blob/master/src/facenet.py)
- [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
- [tensorflow将训练好的模型freeze,即将权重固化到图里面,并使用该模型进行预测](https://blog.csdn.net/lujiandong1/article/details/53385092)
- [tensorflow，使用freeze_graph.py将模型文件和权重数据整合在一起并去除无关的Op](https://blog.csdn.net/czq7511/article/details/72452985)
- [NVIDIA-Jetson/tf_trt_models](https://github.com/NVIDIA-Jetson/tf_trt_models)使用tensorflow中的tensorrt
- [NVIDIA-Jetson/tf_to_trt_image_classification](https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification)使用Nvidia tensorrt

---
[toc]

# tf model to tensorRT
参考：https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/workflows/tf_to_tensorrt.html

训练`tensorflow`模型直接将模型及参数保存成`tensorrt`格式文件

```python
# -*- coding:utf-8 -*-
'''
参考：
https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/workflows/tf_to_tensorrt.html

将tensorflow model保存成tensorrt格式
'''
import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
# We use PyCUDA to transfer data to/from the GPU and NumPy to store data.
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint # generate a random test case
from PIL import Image
from matplotlib.pyplot import imshow # To show test case
import time
import os

# 最后，我们需要导入UFF工具包，将图形从序列化的冻结TensorFlow模型转换为UFF。
import uff

# Converting the TensorFlow Model to UFF

# 我们现在可以将模型转换为序列化的UFF模型。
# 要转换模型，我们需要至少向uff.from_tensorflow函数提供模型流和所需输出节点的名称。
# UFF Toolkit还包含一个uff.from_tensorflow_frozen_model函数，
# 可以从TensorFlow冻结的protobuf文件创建UFF模型。 这些功能有以下选项：
'''
quiet 来抑制转换日志记录
input_nodes 允许您在图中定义一组输入节点（默认值为占位符节点）
text 将让您保存UFF模型的人类可读版本以及二进制UFF
list_nodes 将列出图中的节点
除了返回序列化模型外，output_filename还会将模型写入指定的文件路径
'''

# --------------training a model in tensorflow
STARTER_LEARNING_RATE = 1e-4
BATCH_SIZE = 10
NUM_CLASSES = 10
MAX_STEPS = 3000
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2
OUTPUT_NAMES = ["fc2/Relu"]

def WeightsVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, name='weights'))

def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='biases'))

def Conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    filter_size = W.get_shape().as_list()
    pad_size = filter_size[0]//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    x = tf.pad(x, pad_mat)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def MaxPool2x2(x, k=2):
    # MaxPool2D wrapper
    pad_size = k//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def network(images):
    # Convolution 1
    with tf.name_scope('conv1'):
        weights = WeightsVariable([5,5,1,32])
        biases = BiasVariable([32])
        conv1 = tf.nn.relu(Conv2d(images, weights, biases))
        pool1 = MaxPool2x2(conv1)

    # Convolution 2
    with tf.name_scope('conv2'):
        weights = WeightsVariable([5,5,32,64])
        biases = BiasVariable([64])
        conv2 = tf.nn.relu(Conv2d(pool1, weights, biases))
        pool2 = MaxPool2x2(conv2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Fully Connected 1
    with tf.name_scope('fc1'):
        weights = WeightsVariable([7 * 7 * 64, 1024])
        biases = BiasVariable([1024])
        fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Fully Connected 2
    with tf.name_scope('fc2'):
        weights = WeightsVariable([1024, 10])
        biases = BiasVariable([10])
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases)

    return fc2


def loss_metrics(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='softmax')
    return tf.reduce_mean(cross_entropy, name='softmax_mean')


def training(loss):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE,
                                               global_step,
                                               100000,
                                               0.75,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            summary):

    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        log, correctness = sess.run([summary, eval_correct], feed_dict=feed_dict)
        true_count += correctness
    precision = float(true_count) / num_examples
    tf.summary.scalar('precision', tf.constant(precision))
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    return log

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
    feed_dict = {
        images_pl: np.reshape(images_feed, (-1,28,28,1)),
        labels_pl: labels_feed,
    }
    return feed_dict


def run_training(data_sets):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        logits = network(images_placeholder)
        loss = loss_metrics(logits, labels_placeholder)
        train_op = training(loss)
        eval_correct = evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log",
                                               graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log/validation",
                                            graph=tf.get_default_graph())
        sess.run(init)
        for step in range(MAX_STEPS):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join("/tmp/tensorflow/mnist/log", "model.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                print('Validation Data Eval:')
                log = do_eval(sess,
                              eval_correct,
                              images_placeholder,
                              labels_placeholder,
                              data_sets.validation,
                              summary)
                test_writer.add_summary(log, step)
        # Return sess

        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graphdef,
                                                                    OUTPUT_NAMES)
        return tf.graph_util.remove_training_nodes(frozen_graph)

MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")
tf_model = run_training(MNIST_DATASETS)

uff_model = uff.from_tensorflow(tf_model, ["fc2/Relu"])

# Importing the UFF Model into TensorRT and Building an Engine

# 我们现在有一个UFF模型流，我们可以用它来构建一个TensorRT引擎。 我们首先为TensorRT创建一个记录器。

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
# 接下来，我们创建一个UFF解析器并确定所需的输入和输出节点。

parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (1,28,28), 0) # [chw]格式 (1,28,28)
parser.register_output("fc2/Relu")

# 现在，我们将记录器，解析器，uff模型流和一些设置（最大批量大小和最大工作空间大小）传递给将为我们构建引擎的效用函数。
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)
# 我们现在可以摆脱解析器了。
parser.destroy()

img, label = MNIST_DATASETS.test.next_batch(1)
img = img[0]
# Convert input data to Float32
img = img.astype(np.float32)
label = label[0]

# imshow(img.reshape(28,28))

# 现在，我们从TensorFlow dataloader获得一个测试用例（将其转换为FP32）。
runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

# 接下来，我们在GPU上以及在主机上分配内存以在推理之后保存结果。 这些分配的大小是输入/预期输出*批量大小的大小。
output = np.empty(10, dtype = np.float32)

# Alocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

# 引擎需要绑定（指向GPU内存的指针）。 PyCUDA让我们通过将内存分配的结果转换为整数来做到这一点。
bindings = [int(d_input), int(d_output)]

# 我们创建一个cuda流来运行推理。
stream = cuda.Stream()

# 现在，我们将数据传输到GPU，运行推理，然后将结果传输到主机。

# Transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
# Execute model
context.enqueue(1, bindings, stream.handle, None)
# Transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
# Syncronize threads
stream.synchronize()

print("Test Case: " + str(label))
print ("Prediction: " + str(np.argmax(output)))

# 我们还可以将引擎保存到文件中以供日后使用。
trt.utils.write_engine_to_file("./tf_mnist.engine", engine.serialize())

# 您可以稍后使用tensorrt.utils.load_engine加载此引擎。
new_engine = trt.utils.load_engine(G_LOGGER, "./tf_mnist.engine")
# 最后，我们清理我们的上下文，引擎和运行时。
context.destroy()
engine.destroy()
new_engine.destroy()
runtime.destroy()
```

# freeze graph to tensorRT
## 1、tf model保存成冻结网络

```python
# -*- coding:utf-8 -*-
'''
参考：
https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/workflows/tf_to_tensorrt.html

将tensorflow model保存成freeze_graph.pb
'''
import tensorflow as tf
import numpy as np
import time
import os

# 最后，我们需要导入UFF工具包，将图形从序列化的冻结TensorFlow模型转换为UFF。
import uff

# Converting the TensorFlow Model to UFF

# --------------training a model in tensorflow------------------
STARTER_LEARNING_RATE = 1e-4
BATCH_SIZE = 10
NUM_CLASSES = 10
MAX_STEPS = 3000
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2
OUTPUT_NAMES = ["fc2/Relu"]

def WeightsVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, name='weights'))

def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='biases'))

def Conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    filter_size = W.get_shape().as_list()
    pad_size = filter_size[0]//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    x = tf.pad(x, pad_mat)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def MaxPool2x2(x, k=2):
    # MaxPool2D wrapper
    pad_size = k//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def network(images):
    # Convolution 1
    with tf.name_scope('conv1'):
        weights = WeightsVariable([5,5,1,32])
        biases = BiasVariable([32])
        conv1 = tf.nn.relu(Conv2d(images, weights, biases))
        pool1 = MaxPool2x2(conv1)

    # Convolution 2
    with tf.name_scope('conv2'):
        weights = WeightsVariable([5,5,32,64])
        biases = BiasVariable([64])
        conv2 = tf.nn.relu(Conv2d(pool1, weights, biases))
        pool2 = MaxPool2x2(conv2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Fully Connected 1
    with tf.name_scope('fc1'):
        weights = WeightsVariable([7 * 7 * 64, 1024])
        biases = BiasVariable([1024])
        fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Fully Connected 2
    with tf.name_scope('fc2'):
        weights = WeightsVariable([1024, 10])
        biases = BiasVariable([10])
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases)

    return fc2


def loss_metrics(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='softmax')
    return tf.reduce_mean(cross_entropy, name='softmax_mean')


def training(loss):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE,
                                               global_step,
                                               100000,
                                               0.75,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            summary):

    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        log, correctness = sess.run([summary, eval_correct], feed_dict=feed_dict)
        true_count += correctness
    precision = float(true_count) / num_examples
    tf.summary.scalar('precision', tf.constant(precision))
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    return log

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
    feed_dict = {
        images_pl: np.reshape(images_feed, (-1,28,28,1)),
        labels_pl: labels_feed,
    }
    return feed_dict


def run_training(data_sets):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        logits = network(images_placeholder)
        loss = loss_metrics(logits, labels_placeholder)
        train_op = training(loss)
        eval_correct = evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log",
                                               graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log/validation",
                                            graph=tf.get_default_graph())
        sess.run(init)
        for step in range(MAX_STEPS):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join("/tmp/tensorflow/mnist/log", "model.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                print('Validation Data Eval:')
                log = do_eval(sess,
                              eval_correct,
                              images_placeholder,
                              labels_placeholder,
                              data_sets.validation,
                              summary)
                test_writer.add_summary(log, step)
        # Return sess

        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graphdef,
                                                                    OUTPUT_NAMES)
        output_graph='./model/freeze_graph.pb'
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        print("%d ops in the final graph." % len(frozen_graph.node))
        # return tf.graph_util.remove_training_nodes(frozen_graph)

MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")

run_training(MNIST_DATASETS)
```

## 2、将冻结网络转成engine格式(使用tensorrt)

```python
# -*- coding:utf-8 -*-
'''
参考：
https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/workflows/tf_to_tensorrt.html

将tensorflow model保存成tensorrt格式
'''
import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
# We use PyCUDA to transfer data to/from the GPU and NumPy to store data.
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint # generate a random test case
from PIL import Image
from matplotlib.pyplot import imshow # To show test case
import time
import os

# 最后，我们需要导入UFF工具包，将图形从序列化的冻结TensorFlow模型转换为UFF。
import uff

# ------freeze graph to trt format(.engine)-----


MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")

freeze_graph='./model/freeze_graph.pb'

# uff_model = uff.from_tensorflow(tf_model, ["fc2/Relu"])
uff_model=uff.from_tensorflow_frozen_model(freeze_graph,["fc2/Relu"])

# Importing the UFF Model into TensorRT and Building an Engine

# 我们现在有一个UFF模型流，我们可以用它来构建一个TensorRT引擎。 我们首先为TensorRT创建一个记录器。

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
# 接下来，我们创建一个UFF解析器并确定所需的输入和输出节点。

parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (1,28,28), 0) # [chw]格式 (1,28,28)
parser.register_output("fc2/Relu")

# 现在，我们将记录器，解析器，uff模型流和一些设置（最大批量大小和最大工作空间大小）传递给将为我们构建引擎的效用函数。
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)
# 我们现在可以摆脱解析器了。
parser.destroy()

img, label = MNIST_DATASETS.test.next_batch(1)
img = img[0]
# Convert input data to Float32
img = img.astype(np.float32)
label = label[0]

# imshow(img.reshape(28,28))

# 现在，我们从TensorFlow dataloader获得一个测试用例（将其转换为FP32）。
runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

# 接下来，我们在GPU上以及在主机上分配内存以在推理之后保存结果。 这些分配的大小是输入/预期输出*批量大小的大小。
output = np.empty(10, dtype = np.float32)

# Alocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

# 引擎需要绑定（指向GPU内存的指针）。 PyCUDA让我们通过将内存分配的结果转换为整数来做到这一点。
bindings = [int(d_input), int(d_output)]

# 我们创建一个cuda流来运行推理。
stream = cuda.Stream()

# 现在，我们将数据传输到GPU，运行推理，然后将结果传输到主机。

# Transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
# Execute model
context.enqueue(1, bindings, stream.handle, None)
# Transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
# Syncronize threads
stream.synchronize()

print("Test Case: " + str(label))
print ("Prediction: " + str(np.argmax(output)))

# 我们还可以将引擎保存到文件中以供日后使用。
trt.utils.write_engine_to_file("./model/tf_mnist.engine", engine.serialize())

# 您可以稍后使用tensorrt.utils.load_engine加载此引擎。
new_engine = trt.utils.load_engine(G_LOGGER, "./model/tf_mnist.engine")
# 最后，我们清理我们的上下文，引擎和运行时。
context.destroy()
engine.destroy()
new_engine.destroy()
runtime.destroy()
```


