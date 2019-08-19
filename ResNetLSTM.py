#coding:utf-8
import torch
import resnet18_feature_extract as rfe
import os
import numpy as np
import random
from datetime import datetime
import threading as th
import time
from PIL import ImageGrab

def GetBatch(data,label,sampleNum=8,batchnum=8):
    '''
    yield关键字，是一个迭代器
    每次按照batch size进行移动
    返回样本
    '''
    for i in range(0,sampleNum//batchnum):
        low=i*batchnum
        x=data[:,low:low+batchnum,:]
        y=label[low:low+batchnum,:]
        yield x,y

class net:
    def __init__(self,hidden=100,lr=0.001):
        self.n1=torch.nn.LSTM(4608,hidden,2)
        #feature number,hidden units,layers

        self.n2=torch.nn.Linear(hidden,1)
        #输出的线性映射层，输出只有一维所以后一个参数为1

        self.criteria=torch.nn.MSELoss()
        #误差是MSE即mean squared error

        self.opt=torch.optim.Adam([{'params':self.n1.parameters()},
                                   {'params':self.n2.parameters()}],lr)
        #训练方法adam,需要训练的参数有n1:LSTM,n2:输出的线性映射层的参数
        #lr为学习率

        self.data=None
        self.label=None
        self.ex=rfe.resnet18_feature_extract()

    def __GetDataSet1(self,dirpath=None):
        '''
        训练时获取样本
        放置在当前目录的sample文件夹下
        或者也可以通过dirpath指定任何一个目录的sample文件夹下
        '''
        if dirpath is None:
            dirpath=os.path.dirname(__file__)+os.sep+'sample'
        ex=self.ex
        #os.sep是路径分割符，windows是双斜杠\\,linux下又不同
        #为了兼容性因此写了os.sep

        sampleNum=0
        data=torch.Tensor()
        #空tensor，用来做之后的叠加

        label=[]
        print('开始加载')
        for dname in os.listdir(dirpath):
            #os.listdir 获取文件夹下所有文件/目录名
            sPath=dirpath+os.sep+dname
            sampleNum+=1
            f=[]
            for j in range(1,30+1):
                #读取样本下的图片
                #o (1).jpg ~ o(30).jpg
                #range(1,30+1)即1~30，注意要加一

                imgPath=sPath+os.sep+'o ({}).jpg'.format(j)
                out=ex.GetFeature(imgPath)
                #提取特征
                #out[1,512,3,3]
                out=out.flatten()
                out=out.unsqueeze(0)
                #铺平[4608]
                #unsqueeze是增加一维，在dim=0的位置
                #[1,4608]
                f.append(out)
            
            with torch.no_grad():
                #30张图片读完后进行叠加，获得样本f
                #f原本只是个list，用stack转化成tensor
                f=torch.stack(f)
                #f.size=[30,1,4608]
                
                data=torch.cat((data,f),dim=1)
                #data则是存放所有样本的变量
                #每一次循环将自己和f进行连接concatenate(cat)
                #连接的位置在第二维，dim=1,(0,1,2)

            
            labelPath=sPath+os.sep+'label.txt'
            tx=open(labelPath)
            str1=tx.read()
            tx.close()
            #读取label.txt中的标签
            label.append([float(str1)])
            #注意使用float() 把str转化为float
            #而且注意方括号[],如果没有方括号
            #label(samples)
            #而不是(samples,features),会无法放到LSTM中训练

            print('sample{} loaded'.format(sampleNum))
        
        label=np.array(label)
        #label=self.scaler.fit_transform(label)
        label=label/15
        #缩小到0~1
        #这里的15需要改动，取决于训练数据中标签的值
        
        label=torch.Tensor(label)

        #shuffle
        indices=np.arange(sampleNum)
        np.random.shuffle(indices)
        data=data[:,indices,:]
        label=label[indices,:]
        return data,label
    def __GetDataSet2(self,sampleDir):
        '''
        测试时获取sampleDir中的样本
        由于只有一个
        所以也不需要os.listdir
        '''
        ex=self.ex
        sPath=sampleDir
        data=torch.Tensor()
        print('开始加载')
        f=[]
        for j in range(1,30+1):
            imgPath=sPath+os.sep+'o ({}).jpg'.format(j)
            out=ex.GetFeature(imgPath)
            out=out.flatten()
            out=out.unsqueeze(0)
            f.append(out)
        with torch.no_grad():
            f=torch.stack(f)
            data=torch.cat((data,f),dim=1)
        print('加载成功')
        return data
    def load(self,saveName):
        '''
        读取当前目录下save中的存档
        存档名为saveName
        '''
        savedir=os.path.dirname(__file__)+os.sep+'save'+os.sep
        savePath=savedir+saveName
        checkpoint = torch.load(savePath)
        self.n1.load_state_dict(checkpoint['net1'])
        self.n2.load_state_dict(checkpoint['net2'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        #start_epoch = checkpoint['epoch'] + 1

    def loadData(self,dataPath=None):
        '''
        读取datapath下的数据
        放到类的全局变量data，label中
        省去重复读取样本的时间
        '''
        self.data,self.label=self.__GetDataSet1(dataPath)

    def train(self,epochNum=100,batchNum=8,finalLoss=1e-5):
        '''
        epochNum:训练的次数
        batchNum:一批次中的样本数量，必须小于样本数量*0.8
        finalLoss:指定到达多小的误差后就存档退出
        '''
        if self.data is None:
            #如果没有通过loadData读过样本
            #就从默认路径读取
            data,label=self.__GetDataSet1()
            self.data=data
            self.label=label
        else:
            #读取过就直接搞过来
            #省去重复读取
            data=self.data
            label=self.label
        sampleNum=label.size()[0]
        #通过获取tensor的大小来得知文件夹中样本的个数
        num_test=int(0.2*sampleNum)
        #把样本按照8:2分成训练集和测试集
        #可以改的更小一些，如0.005等等
        #影响效率，因为每一epoch测试集误差是通过计算所有的测试样本得到的

        train_input = data[:,num_test:,:]
        train_output = label[num_test:,:]
        test_input = data[:,:num_test,:]
        test_output = label[:num_test,:]
        trainNum=sampleNum-num_test

        self.n1.train()
        self.n2.train()
        #调成训练模式
        
        print('train')

        savedir=os.path.dirname(__file__)+os.sep+'save'+os.sep
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for epoch in range(epochNum):
            train_loss=0
            test_loss=0
            for x,y in GetBatch(train_input,train_output,
                                trainNum,batchNum):
                #x[30,batch,4608]
                #y[batch,1]
                self.opt.zero_grad()
                #清空之前训练的梯度
                #因为RNN的特性，如果不清空就变成了30+30+++长的序列
                
                out,_=self.n1(x)
                #返回两个out和cell
                #由于cell没用到就用下划线忽略
                #实际上是在n1.parameters中已经获取过了

                #out[30,batch,1000]
                out_last=out[-1,:,:]
                #out_last[batch,1000]
                #-1获取序列中最后一次的输出

                pred=self.n2(out_last)
                #pred[batch,1]
                #线性层获得输出

                loss=torch.sqrt(self.criteria(pred,y))
                #开方一下,把MSE改成RMSE，更加的直观
                #乘以15以后就能反应实际预测的偏差

                loss.backward()
                self.opt.step()
                #反向传播误差并进行一次调优训练

                train_loss+=loss.item()
                #print('batch loss',loss.item())
            train_loss/=trainNum//batchNum
            #得到训练时平均每个样本的误差

            #test
            with torch.no_grad():
                #把所有的训练集放进去计算
                #得到RMSE
                #因此训练集太多可能会影响效率
                
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
                #按当前时间指定文件名
                torch.save(state,savedir+saveName)
                if test_loss<finalLoss:
                    break
    def eval(self,samplePath,savePath=None):
        self.n1.eval()
        self.n2.eval()
        #切换到评估模式
        sample=self.__GetDataSet2(samplePath)
        if savePath is not None:
            self.load(savePath)
        with torch.no_grad():
            out,_=self.n1(sample)
            out_last=out[-1,:,:]
            pred=self.n2(out_last)
        #pred=self.scaler.inverse_transform(pred)
        pred=pred*15
        #predx15，反归一化后得到预测的结果
        pred=pred.data.cpu().numpy()[0][0]
        labelPath=samplePath+os.sep+'label.txt'
        tx=open(labelPath)
        str1=tx.read()
        #读取label中的ground truth一起显示
        print('pred:{0},truth:{1}'.format(pred,str1))
        #return pred

if __name__=='__main__':
    n=net()
    n.loadData()
