参考：https://blog.csdn.net/wspba/article/details/56019373

# tensorflow model
- https://github.com/tensorflow/models/tree/master/research/slim/nets

# ResNet
![这里写图片描述](http://img.blog.csdn.net/20171129220119858?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Residual block(basic)

![这里写图片描述](https://img-blog.csdn.net/20170220201128938?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3NwYmE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

# deeper的residual block

![这里写图片描述](https://img-blog.csdn.net/20170220210432796?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3NwYmE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


- 描述Residual的结构与deeper的residual block

![这里写图片描述](https://github.com/raghakot/keras-resnet/raw/master/images/residual_block.png?raw=true)


对于shortcut的方式，作者提出了三个选项：

A. 使用恒等映射，如果residual block的输入输出维度不一致，对增加的维度用0来填充；

B. 在block输入输出维度一致时使用恒等映射，不一致时使用线性投影以保证维度一致；

C. 对于所有的block均使用线性投影。

对这三个选项都进行了实验，发现虽然C的效果好于B的效果好于A的效果，但是差距很小，因此线性投影并不是必需的，而使用0填充时，可以保证模型的复杂度最低，这对于更深的网络是更加有利的。


![这里写图片描述](https://img-blog.csdn.net/20170220205131506?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3NwYmE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


# ResNet50和ResNet101
![这里写图片描述](https://img-blog.csdn.net/20180114205444652?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)