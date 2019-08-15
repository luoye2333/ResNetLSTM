import torch
import torchvision as tv
import cv2.cv2 as cv2
import os
import numpy as np

class resnet18_feature_extract:
    '''
    去掉最后的几层bn,fc,softmax
    获得最后一次卷积后的结果(batch,512,7,7)
    由于relu，其中有很多0，再加一层maxpool
    size=[3,3],stride=2
    得到图片的特征(batch,512,3,3)
    '''
    def __init__(self):
        self.__net=tv.models.resnet18(pretrained=True)
        self.__net.eval()#切换到评估模式
        self.__final_pool=torch.nn.MaxPool2d(3,2)

    def __img_pr(self,img):
        #图片预处理
        #保证和pytorch resnet18训练时一致
        transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
            ])
        img=transform(img)
        img=torch.autograd.Variable(
            torch.unsqueeze(img,dim=0),requires_grad=True
            )
        return img

    def __getfeature(self,input):
        n=self.__net
        pool=self.__final_pool
        #通过层层传递得到输出
        with torch.no_grad():
            x=n.conv1(input)
            x=n.bn1(x)
            x=n.relu(x)
            x=n.maxpool(x)
            x=n.layer1(x)
            x=n.layer2(x)
            x=n.layer3(x)
            x=n.layer4(x)
            x=pool(x)
        return x
        #转化为ndarray
        #return x.data.cpu().numpy()

    def GetFeature(self,imgPath):
        img=cv2.imread(imgPath)
        if img is None:
            raise Exception('Load image@{} failed'.format(imgPath))
        img=self.__img_pr(img)
        return self.__getfeature(img)

    def GetFeature2(self,img):
        img=self.__img_pr(img)
        return self.__getfeature(img)

if __name__=='__main__':
    path=os.path.dirname(__file__)
    img_path=path+os.sep+'o (1).jpg'
    img=cv2.imread(img_path)
    if img is None:
        raise Exception('image')
    img=img_pr(img)
    print(getfeature(n,img).size())

#也可以通过hook获取
#但有冗余计算
#def fhook(self,input,output):
#    global f
#    f=output.data.cpu().numpy()
#n.layer4.register_forward_hook(fhook)
