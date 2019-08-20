#coding:utf-8
import torch
import torchvision as tv
import os
import numpy as np
import random
import cv2.cv2 as cv2
from datetime import datetime
import time

def GetBatch(data,label,sampleNum=8,batchnum=8):
    for i in range(0,sampleNum//batchnum):
        low=i*batchnum
        x=data[low:low+batchnum]
        y=label[low:low+batchnum]
        yield x,y

class net:
    def __init__(self,hidden=100,lr=0.001):
        features=4608
        layers=2
        output=1
        self.frames=30
        self.sampleNum=-1

        self.cnn=tv.models.resnet18(pretrained=True)
        self.cnn.eval()
        self.final_pool=torch.nn.MaxPool2d(3,2)

        self.LSTM=torch.nn.LSTM(features,hidden,layers,batch_first=True)
        self.Linear=torch.nn.Linear(hidden,output)
        self.criteria=torch.nn.MSELoss()
        self.opt=torch.optim.Adam([{'params':self.LSTM.parameters()},
                                   {'params':self.Linear.parameters()}],lr)
        self.data=None
        self.label=None

    def loadData(self,samplePath=None):
        self.picRead(samplePath)
        self.normalize()
        self.extractFeature()
        self.shuffle()

    def picRead(self,dirpath=None):
        if dirpath is None:
            dirpath=os.path.dirname(__file__)+os.sep+'sample'

        st=time.time()
        data=[]
        label=[]
        sampleNum=0
        for sname in os.listdir(dirpath):
            spath=dirpath+os.sep+sname
            frames=[]
            for i in range(1,self.frames+1):
                imgname='o ({}).jpg'.format(i)
                img=cv2.imread(spath+os.sep+imgname)
                frames.append(img)
            data.append(frames)

            labelPath=spath+os.sep+'label.txt'
            tx=open(labelPath)
            str1=tx.read()
            tx.close()
            label.append([float(str1)])

            sampleNum+=1
            print('sample{} finished'.format(sampleNum))
        print('sample loaded,time:{:.2f}s'.format(time.time()-st))
        self.sampleNum=sampleNum
        self.data=np.array(data)
        self.label=np.array(label)

    def normalize(self):
        data=self.data
        label=self.label

        st=time.time()
        sampleNum=self.sampleNum
        frames=self.frames
        ndata=torch.zeros(sampleNum,frames,3,224,224)
        for s in range(sampleNum):
            for f in range(frames):
                img=data[s][f]
                transform=tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
                    ])
                img=transform(img)
                img=torch.autograd.Variable(img,requires_grad=True)
                ndata[s][f]=img

        nlabel=label/15#缩小到0~1
        nlabel=torch.Tensor(nlabel)

        print('normalization finished,time:{:.2f}s'.format(time.time()-st))
        
        self.data=ndata
        self.label=nlabel

    def extractFeature(self):
        st=time.time()

        n=self.cnn
        pool=self.final_pool
        data=self.data
        sampleNum=self.sampleNum
        frames=self.frames

        ndata=torch.zeros(sampleNum,frames,4608)
        with torch.no_grad():
            for i in range(sampleNum):
                input=data[i]
                x=n.conv1(input)
                x=n.bn1(x)
                x=n.relu(x)
                x=n.maxpool(x)
                x=n.layer1(x)
                x=n.layer2(x)
                x=n.layer3(x)
                x=n.layer4(x)
                x=pool(x)
                x=x.flatten(start_dim=1)
                ndata[i]=x
        self.data=ndata
        print('feature extracted,time:{:.2f}s'.format(time.time()-st))

    def shuffle(self):
        st=time.time()
        indices=np.arange(self.sampleNum)
        np.random.shuffle(indices)
        self.data=self.data[indices]
        self.label=self.label[indices]
        print('shuffle,time:{:.2f}s'.format(time.time()-st))
        
    def train(self,epochNum=100,batchNum=8,finalLoss=1e-5):
        if self.data is None:
            self.loadData()
        else:
            data=self.data
            label=self.label

        sampleNum=self.sampleNum
        num_test=int(0.2*sampleNum)
        train_input = data[num_test:]
        train_output = label[num_test:]
        test_input = data[:num_test]
        test_output = label[:num_test]
        trainNum=sampleNum-num_test
        if trainNum<batchNum:
            raise Exception('样本太少，或减少batch size')

        self.LSTM.train()
        self.Linear.train()
        
        print('train')

        savedir=os.path.dirname(__file__)+os.sep+'save'+os.sep
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for epoch in range(epochNum):
            train_loss=0
            test_loss=0
            for x,y in GetBatch(train_input,train_output,
                                trainNum,batchNum):
                
                self.opt.zero_grad()
                out,_=self.LSTM(x)
                out_last=out[:,-1,:]
                pred=self.Linear(out_last)
                loss=torch.sqrt(self.criteria(pred,y))
                loss.backward()
                self.opt.step()

                train_loss+=loss.item()

            train_loss/=trainNum//batchNum

            #test loss
            with torch.no_grad():
                out,_=self.LSTM(test_input)
                out_last=out[:,-1,:]
                pred=self.Linear(out_last)
                test_loss=torch.sqrt(self.criteria(pred,test_output))

            print('epoch:{},train:{},test:{}'.format(
                epoch,train_loss,test_loss))

            if (epoch%5==0)or(test_loss<finalLoss):
                state = {'net1':self.LSTM.state_dict(),
                         'net2':self.Linear.state_dict(),
                         'optimizer':self.opt.state_dict()}
                saveName='{}.pth'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                torch.save(state,savedir+saveName)

                if test_loss<finalLoss:
                    break

    def eval(self,samplePath):
        self.LSTM.eval()
        self.Linear.eval()
        with torch.no_grad():
            print('开始加载')
            sample=torch.zeros(30,3,224,224)
            for j in range(1,self.frames+1):
                imgPath=samplePath+os.sep+'o ({}).jpg'.format(j)
                img=cv2.imread(imgPath)
                img=self.__preprocess(img)
                sample[j-1]=img
            sample=self.__getFeature(sample)
            sample=sample.flatten(start_dim=1)
            sample=sample.unsqueeze(dim=0)
            print('加载成功')

            out,_=self.LSTM(sample)
            out_last=out[-1,:,:]
            pred=self.Linear(out_last)

        pred=pred*15
        pred=pred.data.cpu().numpy()[0][0]
        labelPath=samplePath+os.sep+'label.txt'
        tx=open(labelPath)
        str1=tx.read()
        print('pred:{0},truth:{1}'.format(pred,str1))

    def load(self,saveName):
        savedir=os.path.dirname(__file__)+os.sep+'save'
        savePath=savedir+os.sep+saveName
        checkpoint = torch.load(savePath)
        self.LSTM.load_state_dict(checkpoint['net1'])
        self.Linear.load_state_dict(checkpoint['net2'])
        self.opt.load_state_dict(checkpoint['optimizer'])

    def __preprocess(self,img):
        transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
            ])
        img=transform(img)
        img=torch.autograd.Variable(img,requires_grad=True)
        return img
    def __getFeature(self,input):
        n=self.cnn
        pool=self.final_pool
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

if __name__=='__main__':
    n=net()
    n.loadData()
    n.train()