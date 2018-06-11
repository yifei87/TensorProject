- [DenseNet算法详解](https://blog.csdn.net/u014380165/article/details/75142664)
- [DenseNet](https://github.com/liuzhuang13/DenseNet)
- https://github.com/taki0112/Densenet-Tensorflow/


先列下DenseNet的几个优点，感受下它的强大：
- 1、减轻了vanishing-gradient（梯度消失）
- 2、加强了feature的传递
- 3、更有效地利用了feature
- 4、一定程度上较少了参数数量
- 注：每个conv对应的顺序为 BN-Relu-conv(一般为conv-BN-Relu or conv-Relu-BN(dropout))



那就是在保证网络中层与层之间最大程度的信息传输的前提下，直接将所有层连接起来！

先放一个dense block的结构图。在传统的卷积神经网络中，如果你有L层，那么就会有L个连接，但是在DenseNet中，会有L(L+1)/2个连接。简单讲，就是每一层的输入来自前面所有层的输出。如下图：x0是input，H1的输入是x0（input），H2的输入是x0和x1（x1是H1的输出）……

![这里写图片描述](https://img-blog.csdn.net/20170715081827257?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

前面提到过梯度消失问题在网络深度越深的时候越容易出现，原因就是输入信息和梯度信息在很多层之间传递导致的，而现在这种dense connection相当于每一层都直接连接input和loss，因此就可以减轻梯度消失现象，这样更深网络不是问题。另外作者还观察到这种dense connection有正则化的效果，因此对于过拟合有一定的抑制作用，博主认为是因为参数减少了（后面会介绍为什么参数会减少），所以过拟合现象减轻。

第一个公式是ResNet的。这里的l表示层，xl表示l层的输出，Hl表示一个非线性变换。所以对于ResNet而言，l层的输出是l-1层的输出加上对l-1层输出的非线性变换。

![](https://img-blog.csdn.net/20170715081918000?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

第二个公式是DenseNet的。[x0,x1,…,xl-1]表示将0到l-1层的输出feature map做concatenation。concatenation是做通道的合并，就像Inception那样。而前面resnet是做值的相加，通道数是不变的。Hl包括BN，ReLU和3*3的卷积。

![](https://img-blog.csdn.net/20170715081947337?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

前面的Figure 1表示的是dense block，而下面的Figure 2表示的则是一个DenseNet的结构图，在这个结构图中包含了3个dense block。作者将DenseNet分成多个dense block，原因是希望各个dense block内的feature map的size统一，这样在做concatenation就不会有size的问题。

![](https://img-blog.csdn.net/20170715082015009?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

另外这里每个dense block的3*3卷积前面都包含了一个1*1的卷积操作，就是所谓的bottleneck layer，目的是减少输入的feature map数量，既能降维减少计算量，又能融合各个通道的特征，何乐而不为。另外作者为了进一步压缩参数，在每两个dense block之间又增加了1*1的卷积操作。因此在后面的实验对比中，如果你看到DenseNet-C这个网络，表示增加了这个Translation layer，该层的1*1卷积的输出channel默认是输入channel到一半。如果你看到DenseNet-BC这个网络，表示既有bottleneck layer，又有Translation layer。

![](https://img-blog.csdn.net/20170715082117405?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

