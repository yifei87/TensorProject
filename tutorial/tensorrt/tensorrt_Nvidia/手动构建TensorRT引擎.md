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

<font size=4>

# [手动构建TensorRT引擎](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/workflows/manually_construct_tensorrt_engine.html)
For this example, we will use `PyTorch`.
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from matplotlib.pyplot import imshow # to show test case

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training a Model in PyTorch
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.5
SEED = 1
LOG_INTERVAL = 10
# Enable Cuda
torch.cuda.manual_seed(SEED)

# Dataloader
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader  = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/mnist/data', train=True, download=True,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/mnist/data', train=False,
                   transform=transforms.Compose([
                   transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
    **kwargs)

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)

def train(epoch):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch,
                          batch * len(data),
                          len(train_loader.dataset),
                          100. * batch / len(train_loader),
                          loss.data.item()))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss,
                  correct,
                  len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


for e in range(EPOCHS):
    train(e + 1)
    test(e + 1)

# Converting the Model into a TensorRT Engine
weights = model.state_dict()

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

# --------下下下下下下下下下下下下下下下下下下-------------------------

builder = trt.infer.create_infer_builder(G_LOGGER)


network = builder.create_network()

# Name for the input layer, data type, tuple for dimension
data = network.add_input("data", trt.infer.DataType.FLOAT, (1, 28, 28))
assert(data)

#-------------
conv1_w = weights['conv1.weight'].cpu().numpy().reshape(-1)
conv1_b = weights['conv1.bias'].cpu().numpy().reshape(-1)
conv1 = network.add_convolution(data, 20, (5,5),  conv1_w, conv1_b)
assert(conv1)
conv1.set_stride((1,1))

#-------------
pool1 = network.add_pooling(conv1.get_output(0), trt.infer.PoolingType.MAX, (2,2))
assert(pool1)
pool1.set_stride((2,2))

#-------------
conv2_w = weights['conv2.weight'].cpu().numpy().reshape(-1)
conv2_b = weights['conv2.bias'].cpu().numpy().reshape(-1)
conv2 = network.add_convolution(pool1.get_output(0), 50, (5,5), conv2_w, conv2_b)
assert(conv2)
conv2.set_stride((1,1))

#-------------
pool2 = network.add_pooling(conv2.get_output(0), trt.infer.PoolingType.MAX, (2,2))
assert(pool2)
pool2.set_stride((2,2))

#-------------
fc1_w = weights['fc1.weight'].cpu().numpy().reshape(-1)
fc1_b = weights['fc1.bias'].cpu().numpy().reshape(-1)
fc1 = network.add_fully_connected(pool2.get_output(0), 500, fc1_w, fc1_b)
assert(fc1)

#-------------
relu1 = network.add_activation(fc1.get_output(0), trt.infer.ActivationType.RELU)
assert(relu1)

#-------------
fc2_w = weights['fc2.weight'].cpu().numpy().reshape(-1)
fc2_b = weights['fc2.bias'].cpu().numpy().reshape(-1)
fc2 = network.add_fully_connected(relu1.get_output(0), 10, fc2_w, fc2_b)
assert(fc2)

fc2.get_output(0).set_name("prob")
network.mark_output(fc2.get_output(0))

builder.set_max_batch_size(1)
builder.set_max_workspace_size(1 << 20)

engine = builder.build_cuda_engine(network)
network.destroy()
builder.destroy()

# --------------上上上上上上上上上上上上上上--------------------------

runtime = trt.infer.create_infer_runtime(G_LOGGER)
img, target = next(iter(test_loader))
img = img.numpy()[0]
target = target.numpy()[0]
# %matplotlib inline
img.shape
imshow(img[0])
print("Test Case: " + str(target))
img = img.ravel()

context = engine.create_execution_context()

output = np.empty(10, dtype = np.float32)

# Allocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

# Transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
#execute model
context.enqueue(1, bindings, stream.handle, None)
# Transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
# Synchronize threads
stream.synchronize()

print("Test Case: " + str(target))
print ("Prediction: " + str(np.argmax(output)))

trt.utils.write_engine_to_file("./pyt_mnist.engine", engine.serialize())

new_engine = trt.utils.load_engine(G_LOGGER, "./pyt_mnist.engine")

context.destroy()
engine.destroy()
new_engine.destroy()
runtime.destroy()
```