#coding:utf-8
import torch
import resnet18_feature_extract as rfe
import os
import numpy as np
import random
#from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import threading as th
import time
from PIL import ImageGrab
#import win32con
#import win32api
def GetPoint():
    m=PyMouse()
    return m.position()
def GetBatch(data,label,sampleNum=8,batchnum=8):
    for i in range(0,sampleNum//batchnum):
        low=i*batchnum
        x=data[:,low:low+batchnum,:]
        y=label[low:low+batchnum,:]
        yield x,y

class net:
    def __init__(self,hidden=100,lr=0.001):

        self.n1=torch.nn.LSTM(4608,hidden,2)
        self.n2=torch.nn.Linear(hidden,1)
        self.criteria=torch.nn.MSELoss()
        self.opt=torch.optim.Adam([{'params':self.n1.parameters()},
                                   {'params':self.n2.parameters()}],lr)
        self.data=None
        self.label=None
        self.ex=rfe.resnet18_feature_extract()
        #self.scaler=MinMaxScaler()
        self.image_queue=np.zeros([30,224,224,3])
        self.queue_top=0

    def __GetDataSet1(self,dirpath=None):
        if dirpath is None:
            dirpath=os.path.dirname(__file__)+os.sep+'sample'
        ex=self.ex
        #dirpath='Q:\workplace\code\python\ResNetLSTM\sample'
        sampleNum=0
        data=torch.Tensor()
        label=[]
        print('开始加载')
        for dname in os.listdir(dirpath):
            sPath=dirpath+os.sep+dname
            sampleNum+=1
            f=[]
            for j in range(1,30+1):
                imgPath=sPath+os.sep+'o ({}).jpg'.format(j)
                out=ex.GetFeature(imgPath)
                #[1,512,3,3]
                out=out.flatten()
                out=out.unsqueeze(0)
                #[1,4608]
                f.append(out)
            with torch.no_grad():
                f=torch.stack(f)
                #f.size=[30,1,4608]
                data=torch.cat((data,f),dim=1)
            
            labelPath=sPath+os.sep+'label.txt'
            tx=open(labelPath)
            str1=tx.read()
            tx.close()
            label.append([float(str1)])
            print('sample{} loaded'.format(sampleNum))
        
        label=np.array(label)
        #label=self.scaler.fit_transform(label)
        label=label/15#缩小到0~1
        label=torch.Tensor(label)

        #shuffle
        indices=np.arange(sampleNum)
        np.random.shuffle(indices)
        data=data[:,indices,:]
        label=label[indices,:]
        return data,label
    def __GetDataSet2(self,sampleDir):
        ex=self.ex
        sPath=sampleDir
        data=torch.Tensor()
        print('开始加载')
        f=[]
        for j in range(1,30+1):
            imgPath=sPath+os.sep+'o ({}).jpg'.format(j)
            out=ex.GetFeature(imgPath)
            #[1,512,3,3]
            out=out.flatten()
            out=out.unsqueeze(0)
            #[1,4608]
            f.append(out)
        with torch.no_grad():
            f=torch.stack(f)
            #f.size=[30,1,4608]
            data=torch.cat((data,f),dim=1)
        print('加载成功')
        return data
    def load(self,saveName):
        savedir=os.path.dirname(__file__)+os.sep+'save'+os.sep
        savePath=savedir+saveName
        checkpoint = torch.load(savePath)
        self.n1.load_state_dict(checkpoint['net1'])
        self.n2.load_state_dict(checkpoint['net2'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        #start_epoch = checkpoint['epoch'] + 1

    def loadData(self,dataPath=None):
        self.data,self.label=self.__GetDataSet1(dataPath)

    def train(self,epochNum=100,batchNum=8,finalLoss=1e-5):
        if self.data is None:
            data,label=self.__GetDataSet1()
            self.data=data
            self.label=label
        else:
            data=self.data
            label=self.label
        sampleNum=label.size()[0]
        num_test=int(0.2*sampleNum)
        train_input = data[:,num_test:,:]
        train_output = label[num_test:,:]
        test_input = data[:,:num_test,:]
        test_output = label[:num_test,:]
        trainNum=sampleNum-num_test

        self.n1.train()
        self.n2.train()
        
        print('train')

        savedir=os.path.dirname(__file__)+os.sep+'save'+os.sep
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for epoch in range(epochNum):

            #esc=win32api.GetAsyncKeyState(win32con.VK_ESCAPE)
            #if esc<0:
            #    break

            train_loss=0
            test_loss=0
            for x,y in GetBatch(train_input,train_output,
                                trainNum,batchNum):
                #x[30,batch,4608]
                #y[batch,1]
                self.opt.zero_grad()
                out,_=self.n1(x)
                #out[30,batch,1000]
                out_last=out[-1,:,:]
                #out_last[batch,1000]
                pred=self.n2(out_last)
                #pred[batch,1]
                loss=torch.sqrt(self.criteria(pred,y))
                loss.backward()
                self.opt.step()

                train_loss+=loss.item()
                #print('batch loss',loss.item())
            train_loss/=trainNum//batchNum
            #test
            with torch.no_grad():
                out,_=self.n1(test_input)
                out_last=out[-1,:,:]
                pred=self.n2(out_last)
                test_loss=torch.sqrt(self.criteria(pred,test_output))

            print('epoch:{},train:{},test:{}'.format(
                epoch,train_loss,test_loss))

            if (epoch%5==0)or(test_loss<finalLoss):
                #每5个epoch保存
                #或者达成指标直接保存退出
                state = {'net1':self.n1.state_dict(),
                         'net2':self.n2.state_dict(),
                         'optimizer':self.opt.state_dict(),
                         'epoch':epoch}
                saveName='{}.pth'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                torch.save(state,savedir+saveName)
                if test_loss<finalLoss:
                    break
    def eval(self,samplePath,savePath=None):
        self.n1.eval()
        self.n2.eval()
        sample=self.__GetDataSet2(samplePath)
        if savePath is not None:
            self.load(savePath)
        with torch.no_grad():
            out,_=self.n1(sample)
            out_last=out[-1,:,:]
            pred=self.n2(out_last)
        #pred=self.scaler.inverse_transform(pred)
        pred=pred*15
        pred=pred.data.cpu().numpy()[0][0]
        labelPath=samplePath+os.sep+'label.txt'
        tx=open(labelPath)
        str1=tx.read()
        print('pred:{0},truth:{1}'.format(pred,str1))
        #return pred
'''
    def demo(self,p):
        t=th.Thread(target=__get_queue,name="imageCapture")
        t.start()
        
        
        self.bbox=(p[0],p[1],p[0]+224,p[1]+224)
        self.n1.eval()
        self.n2.eval()
        sample=self.__get_queue()
        with torch.no_grad():
            out,_=self.n1(sample)
            out_last=out[-1,:,:]
            pred=self.n2(out_last)
        #pred=self.scaler.inverse_transform(pred)
        pred=pred*15
        pred=pred.data.cpu().numpy()[0][0]
        labelPath=samplePath+os.sep+'label.txt'
        tx=open(labelPath)
        str1=tx.read()
        print('pred:{0},truth:{1}'.format(pred,str1))
        return pred

    def __get_queue(self):
        samplingRate=15
        frames=30
        while True:
            lastTime=time.time()
            img=ImageGrab.grab(bbox=self.bbox)
            img=np.asarray(img)
            self.image_queue[self.queue_top]=img
            self.queue_top=(self.queue_top+1)%frames
            ntime=time.time()
            #print(ntime)
            time.sleep(1/15-(ntime-lastTime))
'''




if __name__=='__main__':
    n=net()
    n.loadData()
