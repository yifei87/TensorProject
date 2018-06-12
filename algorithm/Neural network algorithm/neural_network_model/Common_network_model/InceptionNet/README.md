[Feature Extractor[Inception v4]](https://www.cnblogs.com/shouhuxianjian/p/7786760.html)
[从Inception v1,v2,v3,v4,RexNeXt到Xception再到MobileNets,ShuffleNet,MobileNetV2](https://blog.csdn.net/qq_14845119/article/details/73648100)

在下面的网络结构图中：所有后面不带V的卷积，用的都是same-padded，也就是输出的网格大小等于输入网格的大小（如vgg的卷积一样）；带V的使用的是valid-padded，表示输出的网格尺寸是会逐步减小的（如lenet5的卷积一样）。

# 1. inception v4

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105125349107-1485454201.png)
**图1.1 inception v4 网络结构图**</center>

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105125306732-1161981138.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105130035185-1509772135.png)
**图1.2 图1.1的stem和Inception-A部分结构图**</center>

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105130154138-78816950.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105130132060-37640786.png)
**图1.3 图1.1的Reduction-A和Inception-B部分结构图**</center>


<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105130200326-420354111.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105130139810-1160495089.png)
**图1.4 图1.1的Reduction-B和Inception-C部分结构图**</center>

# 2. Inception-resnet-v1 & Inception-resnet-v2


<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105131831545-1345841102.jpg)
**图2.1 Inception-resnet-v1 & Inception-resnet-v2的结构图**</center>

## 2.1 Inception-resnet-v1的组成模块
<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105132823170-1060168089.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105132828873-108595857.png)
**图2.1.1 图2.1的stem和Inception-ResNet-A部分结构图**</center>

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105132835420-117438110.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105132840420-727245524.png)
**图2.1.2 图2.1的Reduction-A和Inception-ResNet-B部分结构图**</center>

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105132842685-1296188651.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105132845045-1654562410.png)
**图2.1.3 图2.1的Reduction-B和Inception-ResNet-C部分结构图**</center>

## 2.2 Inception-resnet-v2的组成模块
<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105125306732-1161981138.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105133237810-2131993535.png)
**图2.2.1 图2.1的stem和Inception-ResNet-A部分结构图**</center>

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105133255545-1779348943.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105133259716-799592073.png)
**图2.2.2 图2.1的Reduction-A和Inception-ResNet-B部分结构图**</center>

<center>![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105133302248-1706965340.png)![这里写图片描述](https://images2017.cnblogs.com/blog/441382/201711/441382-20171105133304263-553000587.png)
**图2.2.3 图2.1的Reduction-B和Inception-ResNet-C部分结构图**</center>