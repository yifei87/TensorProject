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

---

# 输出模型文件
## 1、一般使用
```python
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
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Write out 3 files
tf.train.Saver().save(sess, './model/trained.ckpt')
tf.train.write_graph(sess.graph_def, '.', './model/trained.pb', as_text=False) # 这个pb文件只有模型的结构而没有模型参数
tf.train.write_graph(sess.graph_def, '.', './model/trained.txt', as_text=True) # txt文件 显示看到模型的结构

print(sess.run(W))

#'''
# Eval
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#'''

```

## 2、tf.train.Supervisor()
参考： [tensorflow将训练好的模型freeze,即将权重固化到图里面,并使用该模型进行预测](https://blog.csdn.net/lujiandong1/article/details/53385092)

```python
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
	# 所有的图表搭建需在tf.train.Supervisor()之前定义，放在它后定义，运行会报错
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

```

将包含这些目录

```python
.
├── checkpoint
├── trained.ckpt.data-00000-of-00001
├── trained.ckpt.index
├── trained.ckpt.meta
├── trained.pb
└── trained.txt
```
# 冻结网络
转成`pb`文件（包含模型结构和参数）
## 1、使用[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)

具体使用参考[tensorflow，使用freeze_graph.py将模型文件和权重数据整合在一起并去除无关的Op](https://blog.csdn.net/czq7511/article/details/72452985)

```python
# 用法：
python tensorflow/python/tools/freeze_graph.py \
--input_graph=some_graph_def.pb \ 注意：这里的pb文件是用tf.train.write_graph方法保存的
--input_binary=True \ 默认是False 如果出现编码错误，改成True
--input_checkpoint=model.ckpt.1001 \ 注意：这里若是r12以上的版本，只需给.data-00000....前面的文件名，如：model.ckpt.1001.data-00000-of-00001，只需写model.ckpt.1001  
--output_graph=/tmp/frozen_graph.pb \
--output_node_names='softmax' # 输出的节点名
```
具体用法如：
```
python3 freeze_graph.py \
--input_graph=./model/trained.pb \
--input_checkpoint=./model/trained.ckpt \
--input_binary=True \
--output_graph=./model/frozen_graph.pb \
--output_node_names='hypothesis'
```
## 2、python执行freeze_graph.py
`trained.pb`（保存网络结构）
`trained.ckpt`（保存网络参数）
```python
# -*- coding:utf-8 -*-
from tensorflow.python.tools.freeze_graph import freeze_graph

input_graph='./model/trained.pb' # 这里的pb文件是用tf.train.write_graph方法保存的
input_checkpoint='./model/trained.ckpt' # 这里若是r12以上的版本，只需给.data-00000....前面的文件名，
                                # 如：model.ckpt.1001.data-00000-of-00001，只需写model.ckpt.1001

output_graph='./model/frozen_graph.pb'
output_node_names='hypothesis' # 这里的输出节点名为 hypothesis 而不是 softmax（由程序而定）

''' ---上面参数随程序修改，以下参数不用修改---------------------'''

input_saver=""
input_binary=True #False # 如果 False时，出现
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 47: invalid start byte
# input_binary 改成True

restore_op_name="save/restore_all"
filename_tensor_name="save/Const:0"
clear_devices=True
initializer_nodes=''


freeze_graph(
    input_graph=input_graph,
    input_saver=input_saver,
    input_binary=input_binary,
    input_checkpoint=input_checkpoint,
    output_node_names=output_node_names,
    restore_op_name=restore_op_name,
    filename_tensor_name=filename_tensor_name,
    output_graph=output_graph,
    clear_devices=clear_devices,
    initializer_nodes=initializer_nodes,
)
```

## 2.2、python执行freeze_graph.py
使用到的文件只是`ckpt`保存的文件
`trained.ckpt.meta`（保存网络结构）
`trained.ckpt`（保存网络参数）
```python
# -*- coding:utf-8 -*-
from tensorflow.python.tools.freeze_graph import freeze_graph

# input_graph='./model/trained.pb' # 这里的pb文件是用tf.train.write_graph方法保存的
input_graph=None

input_meta_graph='./model/trained.ckpt.meta'

input_checkpoint='./model/trained.ckpt' # 这里若是r12以上的版本，只需给.data-00000....前面的文件名，
                                # 如：model.ckpt.1001.data-00000-of-00001，只需写model.ckpt.1001

output_graph='./model/frozen_graph.pb'
output_node_names='hypothesis' # 这里的输出节点名为 hypothesis 而不是 softmax（由程序而定）

''' ---上面参数随程序修改，以下参数不用修改---------------------'''

input_saver=""
input_binary=True #False # 如果 False时，出现
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 47: invalid start byte
# input_binary 改成True

restore_op_name="save/restore_all"
filename_tensor_name="save/Const:0"
clear_devices=True
initializer_nodes=''


freeze_graph(
    input_graph=input_graph,
    input_meta_graph=input_meta_graph,
    input_saver=input_saver,
    input_binary=input_binary,
    input_checkpoint=input_checkpoint,
    output_node_names=output_node_names,
    restore_op_name=restore_op_name,
    filename_tensor_name=filename_tensor_name,
    output_graph=output_graph,
    clear_devices=clear_devices,
    initializer_nodes=initializer_nodes,
)
```
## 3、参考freeze_graph.py改写
参考[tensorflow将训练好的模型freeze,即将权重固化到图里面,并使用该模型进行预测](https://blog.csdn.net/lujiandong1/article/details/53385092)
```
# -*- coding:utf-8 -*-
import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_folder,output_node_names):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改

    # output_node_names = "Accuracy/predictions"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    # 相当于将参数已经固化在了图当中
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str,default='./model',help="ckpt Model folder to export")
    parser.add_argument("--output_node_names", type=str, default='hypothesis',
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()
    model_folder=args.model_folder
    output_node_names=args.output_node_names
    freeze_graph(model_folder,output_node_names)
```



# 加载冻结网络

## 1、参考[facenet.py](https://github.com/davidsandberg/facenet/blob/master/src/facenet.py)

```python
def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
```
## 2、编写脚本
参考：https://blog.csdn.net/lujiandong1/article/details/53385092

```python
# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

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
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="./model/frozen_graph.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    # 加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name, op.values())
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    # 注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('hypothesis:0')


    with tf.Session(graph=graph) as sess:
        y_ = tf.placeholder(tf.float32, [None, 10])  # true value

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("finish")
```

要获取图中ops的名字和对应的tensor的名字,可用如下的代码：

```python
    # We can list operations
    #op.values() gives you a list of tensors it produces
    #op.name gives you the name
    #输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name,op.values())

```

```python
for op in tf.get_default_graph().get_operations():
    print(op.name)
```

# tensorrt(使用tensorflow中tensorrt模块)加速冻结网络
<font color=#FF00FF size=5>(使用tensorflow自带的tensorrt模块)</font>
参考：[NVIDIA-Jetson/tf_trt_models](https://github.com/NVIDIA-Jetson/tf_trt_models)

注：如果`import tensorflow.contrib.tensorrt as trt` 提示 `No such file or directory`，

1、安装最新的tensorflow（1.7以上）`pip3 install -U tensorflow-GPU==1.8 -i https://pypi.douban.com/simple`

or 

2、如果还是出问题，把[trt_convert.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensorrt/python/trt_convert.py)拷到工程下，执行
`import trt_convert as trt`

## 1、简单版

```python
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

with tf.gfile.GFile(frozen_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

output_names=['hypothesis']
input_names=['x']

trt_graph = trt.create_inference_graph(
    input_graph_def=graph_def,
    outputs=output_names,
    max_batch_size=10000,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16', # 数据精度float16
    minimum_segment_size=50
)

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
```

## 2、合并版

```python
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
```

# 使用Nvidia tensorrt APi加速网络
略 单独写
