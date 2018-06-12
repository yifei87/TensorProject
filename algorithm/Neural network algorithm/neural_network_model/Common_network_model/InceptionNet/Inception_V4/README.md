参考：

- [从Inception v1,v2,v3,v4,RexNeXt到Xception再到MobileNets,ShuffleNet,MobileNetV2](https://blog.csdn.net/qq_14845119/article/details/73648100)
- [Deep Learning-TensorFlow (13) CNN卷积神经网络_ GoogLeNet 之 Inception(V1-V4)](https://blog.csdn.net/diamonjoy_zone/article/details/70576775)
- [深度学习卷积神经网络——经典网络GoogLeNet(Inception V3)网络的搭建与实现](https://blog.csdn.net/loveliuzz/article/details/79135583)
- [Feature Extractor[Inception v4]](https://www.cnblogs.com/shouhuxianjian/p/7786760.html)
- [mobilenet网络的理解](https://blog.csdn.net/wfei101/article/details/78310226)
- [(深度学习)比较新的网络模型：Inception-v3 ， ResNet， Inception-v4， Dual-Path-Net ， Dense-net ， SEnet ， Wide ResNet](https://blog.csdn.net/ling00007/article/details/79115156)

---

参考：

1. Inception[V1]: [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)

2. Inception[V2]: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

3. Inception[V3]: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

4. Inception[V4]: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

---

- [InceptionV4](InceptionV4)
- [Inception-ResNet V1 & Inception-ResNet V2](Inception-ResNet%20V1%20&%20Inception-ResNet%20V2)
- [ResNeXt](ResNeXt)
- [Xception](Xception)
- [MobileNets](MobileNets)
- [ShuffleNet](ShuffleNet)
- [MobileNetV2](MobileNetV2)
- [capnet](capnet)




# InceptionV4

Inception v4主要利用残差连接（Residual Connection）来改进v3结构，代表作为，Inception-ResNet-v1，Inception-ResNet-v2，Inception-v4

resnet中的残差结构如下，这个结构设计的就很巧妙，简直神来之笔，使用原始层和经过2个卷基层的feature map做Eltwise。Inception-ResNet的改进就是使用上文的Inception module来替换resnet shortcut中的conv+1*1 conv。

Inception v4的亮点总结如下：

(1)将Inception模块和ResidualConnection结合，提出了Inception-ResNet-v1，Inception-ResNet-v2，使得训练加速收敛更快，精度更高。

(2)设计了更深的Inception-v4版本，效果和Inception-ResNet-v2相当。

(3)网络输入大小和V3一样，还是299*299



![这里写图片描述](https://img-blog.csdn.net/20180123155046399?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbG92ZWxpdXp6/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Inception-ResNet V1 & Inception-ResNet V2

![这里写图片描述](https://img-blog.csdn.net/20180123155142717?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbG92ZWxpdXp6/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](https://img-blog.csdn.net/20180123155132604?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbG92ZWxpdXp6/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


# ResNeXt

[Aggregated ResidualTransformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)

这篇提出了resnet的升级版。ResNeXt，the next dimension的意思，因为文中提出了另外一种维度cardinality，和channel和space的维度不同，cardinality维度主要表示ResNeXt中module的个数，最终结论

（1）增大Cardinality比增大模型的width或者depth效果更好

（2）与 ResNet 相比，ResNeXt 参数更少，效果更好，结构更加简单，更方便设计

其中，左图为ResNet的一个module,右图为ResNeXt的一个module，是一种split-transform-merge的思想

![这里写图片描述](https://img-blog.csdn.net/20170922164302872)

# Xception

[Xception: DeepLearning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

这篇文章主要在Inception V3的基础上提出了Xception（Extreme Inception），基本思想就是通道分离式卷积（depthwise separable convolution operation）。最终实现了

(1)模型参数有微量的减少，减少量很少，具体如下，

(2)精度较Inception V3有提高，ImageNET上的精度如下，

先说，卷积的操作，主要进行2种变换，

(1)spatial dimensions，空间变换

(2)channel dimension，通道变换

而Xception就是在这2个变换上做文章。Xception与Inception V3的区别如下：

(1)卷积操作顺序的区别

Inception V3是先做`1*1`的卷积，再做`3*3`的卷积，这样就先将通道进行了合并，即通道卷积，然后再进行空间卷积，而Xception则正好相反，先进行空间的`3*3`卷积，再进行通道的`1*1`卷积。

![这里写图片描述](https://img-blog.csdn.net/20170915103536111)

(2)RELU的有无

这个区别是最不一样的，Inception V3在每个module中都有RELU操作，而Xception在每个module中是**没有RELU**操作的。

# MobileNets
[MobileNets: EfficientConvolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

MobileNets其实就是Exception思想的应用。区别就是Exception文章重点在提高精度，而MobileNets重点在压缩模型，同时保证精度。


depthwiseseparable convolutions的思想就是，分解一个标准的卷积为一个depthwise convolutions和一个pointwise convolution。简单理解就是矩阵的因式分解。

![这里写图片描述](https://img-blog.csdn.net/20170915142243480)

# ShuffleNet
[ShuffleNet: AnExtremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)

这篇文章在mobileNet的基础上主要做了1点改进：

mobileNet只做了3*3卷积的deepwiseconvolution，而1*1的卷积还是传统的卷积方式，还存在大量冗余，ShuffleNet则在此基础上，将1*1卷积做了shuffle和group操作，实现了channel shuffle 和pointwise group convolution操作，最终使得速度和精度都比mobileNet有提升。

如下图所示，

(a)是原始的mobileNet的框架，各个group之间相互没有信息的交流。

(b)将feature map做了shuffle操作

(c)是经过channel shuffle之后的结果。

![这里写图片描述](https://img-blog.csdn.net/20170916115907812)

Shuffle的基本思路如下，假设输入2个group，输出5个group

| group 1   | group 2  |

| 1,2,3,4,5  |6,7,8,9,10 |

转化为矩阵为2*5的矩阵

1 2 3 4 5

6 7 8 9 10

转置矩阵，5*2矩阵

1 6

2 7

3 8

4 9

5 10

摊平矩阵

| group 1   | group 2  | group 3   | group 4  | group 5  |

| 1,6           |2,7          |3,8           |4,9          |5,10        |

ShuffleNet Units 的结构如下，

(a)是一个带depthwiseconvolution (DWConv)的bottleneck unit

(b)在(a)的基础上，进行了pointwisegroup convolution (GConv) and channel shuffle

(c)进行了AVG pooling和concat操作的最终ShuffleNetunit

![这里写图片描述](https://img-blog.csdn.net/20170916115918125)

# MobileNetV2
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

主要贡献有2点：

1，提出了逆向的残差结构（Inverted residuals）

由于MobileNetV2版本使用了残差结构，和resnet的残差结构有异曲同工之妙，源于resnet，却和而不同。

![这里写图片描述](https://img-blog.csdn.net/20180420120911778)

由于Resnet没有使用depthwise conv,所以，在进入pointwise conv之前的特征通道数是比较多的，所以，残差模块中使用了0.25倍的降维。而MobileNet v2由于有depthwise conv，通道数相对较少，所以残差中使用 了6倍的升维。

总结起来，2点区别

（1）ResNet的残差结构是0.25倍降维，MobileNet V2残差结构是6倍升维

（2）ResNet的残差结构中3*3卷积为普通卷积，MobileNet V2中3*3卷积为depthwise conv



MobileNet v1，MobileNet v2 有2点区别：

![这里写图片描述](https://img-blog.csdn.net/20180420120934725)

（1）v2版本在进入3*3卷积之前，先进行了1*1pointwise conv升维，并且经过RELU。

（2）1*1卷积出去后，没有进行RELU操作

2，提出了线性瓶颈单元（linear bottlenecks）

Why no RELU？

首选看看RELU的功能。RELU可以将负值全部映射为0，具有高度非线性。下图为论文的测试。在维度比较低2,3的时候，使用RELU对信息的损失是比较严重的。而单维度比较高15,30时，信息的损失是比较少的。

![这里写图片描述](https://img-blog.csdn.net/20180420120950329)

MobileNet v2中为了保证信息不被大量损失，应此在残差模块中去掉最后一个的RELU。因此，也称为线性模块单元。

MobileNet v2网络结构：

![这里写图片描述](https://img-blog.csdn.net/20180420121013639)

其中，t表示通道的扩大系数expansion factor，c表示输出通道数，

n表示该单元重复次数，s表示滑动步长stride

![这里写图片描述](https://img-blog.csdn.net/20180420121034720)

其中bottleneck模块中，stride=1和stride=2的模块分别如上图所示，只有stride=1的模块才有残差结构。



结果：

MobileNet v2速度和准确性都优于MobileNet v1


> references:

> http://iamaaditya.github.io/2016/03/one-by-one-convolution/

> https://github.com/soeaver/caffe-model

> https://github.com/facebookresearch/ResNeXt

> https://github.com/kwotsin/TensorFlow-Xception

> https://github.com/shicai/MobileNet-Caffe https://github.com/shicai/MobileNet-Caffe

> https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

> https://github.com/HolmesShuan/ShuffleNet-An-Extremely-Efficient-CNN-for-Mobile-Devices-Caffe-Reimplementation

> https://github.com/camel007/Caffe-ShuffleNet

> https://github.com/shicai/MobileNet-Caffe

> https://github.com/chinakook/MobileNetV2.mxnet

