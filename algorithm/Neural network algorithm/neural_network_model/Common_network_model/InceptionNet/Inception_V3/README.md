参考：

- [从Inception v1,v2,v3,v4,RexNeXt到Xception再到MobileNets,ShuffleNet,MobileNetV2](https://blog.csdn.net/qq_14845119/article/details/73648100)
- [Deep Learning-TensorFlow (13) CNN卷积神经网络_ GoogLeNet 之 Inception(V1-V4)](https://blog.csdn.net/diamonjoy_zone/article/details/70576775)
- [深度学习卷积神经网络——经典网络GoogLeNet(Inception V3)网络的搭建与实现](https://blog.csdn.net/loveliuzz/article/details/79135583)
---

参考：

1. Inception[V1]: [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)

2. Inception[V2]: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

3. Inception[V3]: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

4. Inception[V4]: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

---


Inception v3网络，主要在v2的基础上，提出了卷积分解（Factorization），代表作是Inceptionv3版本的GoogleNet。

Inception v3的亮点总结如下：

(1) 将`7*7`分解成两个一维的卷积`（1*7,7*1）`，`3*3`也是一样`（1*3,3*1）`，这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，更加精细设计了35*35/17*17/8*8的模块。

(2)增加网络宽度，网络输入从`224*224`变为了`299*299`。

将一个`3x3`卷积拆成`1x3`卷积和`3x1`卷积
![](https://img-blog.csdn.net/20170425211014876?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZGlhbW9uam95X3pvbmU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


![](https://img-blog.csdn.net/20180123155323780?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbG92ZWxpdXp6/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)