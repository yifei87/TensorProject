参考：

- [从Inception v1,v2,v3,v4,RexNeXt到Xception再到MobileNets,ShuffleNet,MobileNetV2](https://blog.csdn.net/qq_14845119/article/details/73648100)
- [Deep Learning-TensorFlow (13) CNN卷积神经网络_ GoogLeNet 之 Inception(V1-V4)](https://blog.csdn.net/diamonjoy_zone/article/details/70576775)

---

参考：

1. Inception[V1]: [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)

2. Inception[V2]: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

3. Inception[V3]: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

4. Inception[V4]: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

---

![这里写图片描述](https://img-blog.csdn.net/20170623151534223)

假设previous layer的大小为28*28*192，则，

a的weights大小，1*1*192*64+3*3*192*128+5*5*192*32=387072

a的输出featuremap大小，28*28*64+28*28*128+28*28*32+28*28*192=28*28*416

b的weights大小，1*1*192*64+(1*1*192*96+3*3*96*128)+(1*1*192*16+5*5*16*32)+1*1*192*32=163328

b的输出feature map大小，28*28*64+28*28*128+28*28*32+28*28*32=28*28*256

写到这里，不禁感慨天才般的1*1 conv，从上面的数据可以看出一方面减少了weights，另一方面降低了dimension。

Inception v1的亮点总结如下：

(1)卷积层共有的一个功能，可以实现通道方向的降维和增维，至于是降还是增，取决于卷积层的通道数（滤波器个数），在Inception v1中1*1卷积用于降维，减少weights大小和feature map维度。

(2)1*1卷积特有的功能，由于1*1卷积只有一个参数，相当于对原始feature map做了一个scale，并且这个scale还是训练学出来的，无疑会对识别精度有提升。

(3)增加了网络的深度

(4)增加了网络的宽度

(5)同时使用了1*1，3*3，5*5的卷积，增加了网络对尺度的适应性

下图为googlenet网络结构：

这里有2个地方需要注意：

(1)整个网络为了保证收敛，有3个loss

(2)最后一个全连接层之前使用的是global average pooling，全局pooling使用的好了，还是有好多地方可以发挥的。

![这里写图片描述](https://img-blog.csdn.net/20180310170208705)

![这里写图片描述](https://img-blog.csdn.net/20170425215006923?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZGlhbW9uam95X3pvbmU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
