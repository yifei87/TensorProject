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

Inception v2的网络，代表作为加入了BN（Batch Normalization）层，并且使用2个3*3替代1个5*5卷积的改进版GoogleNet。

Inception v2的亮点总结如下：

(1)加入了BN层，减少了InternalCovariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯，从而增加了模型的鲁棒性，可以以更大的学习速率训练，收敛更快，初始化操作更加随意，同时作为一种正则化技术，可以减少dropout层的使用。

(2)用2个连续的3*3 conv替代inception模块中的5*5，从而实现网络深度的增加，网络整体深度增加了9层，缺点就是增加了25%的weights和30%的计算消耗。

![这里写图片描述](https://img-blog.csdn.net/20170623151632364)


补充：在 TensorFlow 1.0.0 中采用 **tf.image.per_image_standardization()** 对图像进行标准化，旧版为 **tf.image.per_image_whitening**。

BN 的论文指出，传统的深度神经网络在训练时，每一层的输入的分布都在变化，导致训练变得困难，我们只能使用一个很小的学习速率解决这个问题。而对每一层使用 BN 之后，我们就可以有效地解决这个问题，学习速率可以增大很多倍，达到之前的准确率所需要的迭代次数只有1/14，训练时间大大缩短。而达到之前的准确率后，可以继续训练，并最终取得远超于 Inception V1 模型的性能—— top-5 错误率 4.8%，已经优于人眼水平。因为 BN 某种意义上还起到了正则化的作用，所以可以减少或者取消 Dropout 和 LRN，简化网络结构。


