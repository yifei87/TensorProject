# tensorflow实现的常用模型
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets

# VGG-16
![这里写图片描述](http://img.blog.csdn.net/20171129213446318?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


![这里写图片描述](https://img-blog.csdn.net/20170222092929915?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2N5MTIzNDExODk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


![这里写图片描述](https://img-blog.csdn.net/20170222093054201?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2N5MTIzNDExODk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

'''
VGG是在从Alex-net发展而来的网络。主要修改一下两个方面：
1，在第一个卷基层层使用更小的filter尺寸和间隔（3*3）； 2，在整个图片和multi-scale上训练和测试图片。
3*3 filter:
引入cs231n上面一段话：
几个小滤波器卷积层的组合比一个大滤波器卷积层好：
假设你一层一层地重叠了3个3x3的卷积层（层与层之间有非线性激活函数）。在这个排列下，第一个卷积层中的每个神经元都对输入数据体有一个3x3的视野。
第二个卷积层上的神经元对第一个卷积层有一个3x3的视野，也就是对输入数据体有5x5的视野。同样，在第三个卷积层上的神经元对第二个卷积层有3x3的视野，
也就是对输入数据体有7x7的视野。假设不采用这3个3x3的卷积层，二是使用一个单独的有7x7的感受野的卷积层，那么所有神经元的感受野也是7x7，但是就有一些缺点。
首先，多个卷积层与非线性的激活层交替的结构，比单一卷积层的结构更能提取出深层的更好的特征。其次，假设所有的数据有C个通道，那么单独的7x7卷积层将会包含
7*7*C=49C2个参数，而3个3x3的卷积层的组合仅有个3*（3*3*C）=27C2个参数。直观说来，最好选择带有小滤波器的卷积层组合，而不是用一个带有大的滤波器的卷积层。前者可以表达出输入数据中更多个强力特征，
使用的参数也更少。唯一的不足是，在进行反向传播时，中间的卷积层可能会导致占用更多的内存。
1*1 filter: 作用是在不影响输入输出维数的情况下，对输入线进行线性形变，然后通过Relu进行非线性处理，增加网络的非线性表达能力。 Pooling：2*2，间隔s=2。
'''