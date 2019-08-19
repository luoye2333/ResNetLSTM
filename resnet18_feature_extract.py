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
        #加载预训练的模型

        self.__net.eval()
        #切换到评估模式
        #因为某些层如批量归一化(batch normalization),dropout等
        #训练时和测试时目的功能不一样
        #比如测试时batch size=1,没法进行批量归一化

        self.__final_pool=torch.nn.MaxPool2d(3,2)
        #kernel size,stride
        #方格大小3x3，步长2
        #额外增加一层max pooling
        #去掉多余的0，缩小特征规模
        #Seeing the wind,P4,第二段

    def __img_pr(self,img):
        #图片预处理
        #保证和pytorch resnet18训练时一致
        #https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/torchvision_models.md
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
        #由于去掉fc层比较难
        #所以从头计算到倒数一步也可以
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
        #torch.no_grad()命名空间指定不要计算梯度
        #节约资源，否则内存消耗量飙升

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

