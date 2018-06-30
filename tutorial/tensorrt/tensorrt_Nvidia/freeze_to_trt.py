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