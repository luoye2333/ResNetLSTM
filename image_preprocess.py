import numpy as np
import cv2.cv2 as cv2
class processor:
    def __init__(self,_tw=224,_th=224):
        self.__target_width=_tw
        self.__target_height=_th
    def change_target(self,_tw,_th):
        self.__target_width=_tw
        self.__target_height=_th
    def adjust(self,img):
        #把图片进行缩放处理
        #但不仅仅只是长宽一起缩放
        #而是先根据目标长宽比切割
        #然后缩放
        #保证不失真
        wn,wh=self.__get_slice(img.shape[1],img.shape[0])
        img=self.__slice_img(img,wn,wh)
        img=self.__resize(img)
        return img

    #1：计算原图应该留下的部分
    def __get_slice(self,_width,_height):
        #比较哪个缩放倍数更小
        wscale=_width/self.__target_width
        hscale=_height/self.__target_height
        if wscale<hscale:
            scale=wscale
        else:
            scale=hscale
        #四舍五入到整数
        wn=round(scale*self.__target_width)
        wh=round(scale*self.__target_height)
        return wn,wh

    #2：原图居中，切割掉多余的边缘部分
    def __slice_img(self,img,slicew,sliceh):
        midw=img.shape[1]/2
        midh=img.shape[0]/2
        wl=round(midw-slicew/2)
        wh=round(midw+slicew/2)
        hl=round(midh-sliceh/2)
        hh=round(midh+sliceh/2)
        return img[hl:hh,wl:wh,:]
    
    #3: 缩放
    def __resize(self,img):
        return cv2.resize(img,(self.__target_width,self.__target_height))


if __name__=='__main__':
    pr=processor()
    target_width=224
    target_height=224
    print('image:',end='')
    img_path=input()
    img=cv2.imread(img_path)
    img=pr.adjust(img)
    cv2.imshow('abc',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()