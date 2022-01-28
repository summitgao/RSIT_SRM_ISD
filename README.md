### PyTorch implementation of  [**Remote sensing image translation via style-based recalibration module and improved style discriminator**](https://arxiv.org/pdf/2103.15502.pdf). 

### in *IEEE Geoscience and Remote Sensing Letters*, 2022.

by Tiange Zhang, Feng Gao, Junyu Dong, Qian Du


If you have any questions, please contact us. Email:  zhangtiange@stu.edu.cn, gaofeng@ouc.edu.cn



### Introduction

Existing remote sensing change detection methods are heavily affected by seasonal variation. Since vegetation colors are different between winter and summer, such variations are inclined to be falsely detected as changes. In this letter, we proposed an image translation method to solve the problem. A style-based recalibration module is introduced to capture seasonal features effectively. Then, a new style discriminator is designed to improve the translation performance. The discriminator can not only produce a decision for the fake or real sample, but also return a style vector according to the channel-wise correlations. Extensive experiments are conducted on season-varying dataset. The experimental results show that the proposed method can effectively perform image translation, thereby consistently improving the season-varying image change detection performance



![](https://raw.githubusercontent.com/summitgao/RSIT_SRM_ISD/main/framework.jpg)

The architecture of the basic components from our image-to-image translation network. (a) represents the Generator, which takes an image as the input and outputs the translated image. (b) represents the new Style Discriminator, which is designed to figure out whether the input is a real image or a translated one. It outputs a scalar of [0, 1] as the decision as well as a style vector which encode the style of the input to compute the style loss between real and translated images. (c) illustrates the SRMConvBlock utilized in both Generator and Style Discriminator.

