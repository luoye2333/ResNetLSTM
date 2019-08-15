import time
import numpy as np
from PIL import ImageGrab
from pymouse import PyMouse
#import win32gui, win32ui, win32con, win32api
import cv2.cv2 as cv2
def capture(p1,p2):
    beg = time.time()
    img = ImageGrab.grab(bbox=(p1[0],p1[1],p2[0],p2[1]))
    end = time.time()
    print('time:{}'.format(end - beg))
    return img
def getpoint():
    m=PyMouse()
    return m.position()

def window_capture():
    beg = time.time()
    hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    #saveBitMap.SaveBitmapFile(saveDC, filename)
    rtns = saveBitMap.GetBitmapBits()
    end = time.time()
    print(end - beg)
    return rtns


def releasevideo():
    path=os.getcwd()+'\\picture\\'
    filelist=os.listdir(path)
    fps=15
    size=(1366,768)
    video=cv2.VideoWriter("1.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    for item in filelist:
        if item.endswith('.png'): 
    #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            item = path + item
            img = cv2.imread(item)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()

import resnet18_feature_extract as rfe
class vc:
    def __init__(self):
        self.ex=rfe.resnet18_feature_extract()
    def get(self,p):
        stime=time.time()
        bbox=(p[0],p[1],p[0]+224,p[1]+224)
        for i in range(1,30+1):
            img = ImageGrab.grab(bbox=(p1[0],p1[1],p2[0],p2[1]))
            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            self.ex.GetFeature2(img)

