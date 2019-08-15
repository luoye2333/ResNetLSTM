import pythoncom
import pyHook 
def call1(event): 
    print("Position:", event.Position)
    # 返回 True 可将事件传给其它处理程序，否则停止传播事件
    return True
def main():     
    # 创建钩子管理对象 
    hm = pyHook.HookManager() 
    # 监听所有鼠标事件 
    hm.MouseAllButtonsDown = call1 # 等效于hm.SubscribeMouseAll(OnMouseEvent) 
    # 开始监听鼠标事件 
    hm.HookMouse() 
    # 一直监听，直到手动退出程序 
    pythoncom.PumpMessages()

if __name__ == "__main__": 
    main()