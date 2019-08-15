# ResNetLSTM
use ResNet18 to extract features from a series of image,then feed it into the LSTM network
It can be applied in temporal tasks based on pictures.
For example:
1.Get the waving frequency of a flag,then tell the local wind speed.
*[1] JenniferL.Cardona, MichaelF.Howland, JohnO.Dabiri
Seeing the Wind: Visual Wind Speed Prediction with a Coupled Convolutional and Recurrent Neural Network
See more details at:https://arxiv.org/abs/1905.13290?context=cs.LG*
#### import
import ResNetLSTM as rnl
n=rnl.net()
#### load saved file
n.load('2019-08-14-18-38-43.pth')
#### load dataset
n.loadData('Q:\workplace\code\python\ResNetLSTM\\12345678')
#### train
n.load('2019-08-14-18-38-43.pth')
n.train(epochNum,batchNum,final_loss)

#### evaluate
n.load('2019-08-14-18-38-43.pth')
n.eval('Q:\workplace\code\python\ResNetLSTM\sample\\2019-8-14-14-7-3')

#### preprocess
import image_preprocess as ip
a=ip.processor()
import cv2.cv2 as cv2
img=cv2.imread('o (1).jpg')
img=a.adjust(img)
cv2.imwrite('123.jpg',img)

#### more details
If you want to add some new samples:
1.
create a directory. Name as you want,for the code based on:
for filename in os.listdir()
2.
Put some pictures and a label txt.
can be modified in ResNetLSTM.py line 52 & 89
change the picture name according to ResNetLSTM.py line 53 & 90
picture 'o (1).jpg'
label 'label.txt'
