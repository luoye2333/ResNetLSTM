CSDN博客，稍微详细点的介绍：https://blog.csdn.net/luoye2333/article/details/100182927
## 文件介绍  
ResNetLSTM.py: main source file主程序
image_preprocess.py:  clip images to 224x224把图片预处理成224x224，居中最大尺寸剪切  
sample: put sample folders样本们的母文件夹  
save: store trained parameters放置训练好的参数存档  
eval: sample folders for evaluation同样是放样本用的，但是只用在测试，不用来训练
## 注意点:  
1.样本文件夹下文件名为  
&emsp;&emsp;o (1).jpg  
&emsp;&emsp;label.txt  
2.注意label归一化时的常数是/15  
3.eval函数中有读取label.txt的操作，如果只是测试可以随便放个label.txt或者注释代码也可以  
## !!!!!!!!使用时建议使用交互式!!!!!!  
import ResNetLSTM as rnl  
n=rnl.net()#初始化  
n.loadData()#读取数据集  
n.load('2019-08-14-18-38-43.pth')#读取存档  
n.train()#开始训练  
n.eval ('Q:\workplace\code\python\ResNetLSTM\eval\\2019-8-29-22-36-2')#测试  
#(注意这里路径的2019前多加了一个斜杠是为了转译数字)

在交互式下可以直接取出读取好的数据  
而且出错后不会直接退出  
a,b=n.data,n.label在另一模型使用

更改代码后需要重新加载  
import importlib  
importlib.reload(rnl)  
n.data,n.label=a,b  

附:
图片切割可用同目录下的image_preprocess.py  
注意：居中切割成224x224，小于224x224没有测试过  
import image_preprocess as ip  
a=ip.processor()  
import cv2.cv2 as cv2  
img=cv2.imread('o (1).jpg')  
img=a.adjust(img)  
cv2.imwrite('123.jpg',img)


# ResNetLSTM
use ResNet18 to extract features from a series of image,then feed it into the LSTM network  

It can be applied in temporal tasks based on pictures.  
For example:  
1.Get the waving frequency of a flag,then tell the local wind speed.  
*[1] JenniferL.Cardona, MichaelF.Howland, JohnO.Dabiri.  
Seeing the Wind: Visual Wind Speed Prediction with a Coupled Convolutional and Recurrent Neural Network  
See more details at:https://arxiv.org/abs/1905.13290?context=cs.LG*

### import
import ResNetLSTM as rnl  
n=rnl.net()#get an object  
n.loadData()#load dataset  
n.load('2019-08-14-18-38-43.pth')#load trained parameters  
n.train(epochNum,batchNum,final_loss)  
n.eval ('Q:\workplace\code\python\ResNetLSTM\eval\\2019-8-29-22-36-2')#evaluate  

### preprocess
(file image_preprocess.py clips the image to 224x224 in the center)  

import image_preprocess as ip  
a=ip.processor()  
import cv2.cv2 as cv2  
img=cv2.imread('o (1).jpg')  
img=a.adjust(img)  
cv2.imwrite('123.jpg',img)  

### more details
1.If you want to add some new samples:   
Put some pictures and a label txt.  
&emsp;&emsp;o (1).jpg  
&emsp;&emsp;...  
&emsp;&emsp;o (30).jpg  
&emsp;&emsp;label.txt  
This accords to ResNetLSTM.py line 60, you can change it  

2.line 108, all labels are divided by 15 to normalize, this constant needs to be changed when different dataset is applied.

3.line 256, in eval() function, label.txt is also read in to show the difference betwwen prediction and ground truth. If it is just used to predict, just comment it or put a label.txt with a number as you like.

