"""
功能：手势操作电脑鼠标
1、使用OpenCV读取摄像头视频流；
2、识别手掌关键点像素坐标；
3、根据坐标计算不同的手势模式
4、控制对应的鼠标操作：移动、单击、双击、右击、向上滑、向下滑、拖拽
"""

# 导入OpenCV
import cv2
# 导入handprocess
from gesture_utils.gesture_processor import HandProcess, Utils

# 导入其他依赖包
import time
import numpy as np
import pyautogui
# from gesture_utils.utils import Utils
# import autopy


# 识别控制类
class VirtualMouse:
    def __init__(self):
        
        # image实例，以便另一个类调用
        self.image=None
    
    # 主函数
    def recognize(self):

        handprocess = HandProcess(static_image_mode = False,max_num_hands = 2)
        utils = Utils()
        
        fpsTime = time.time()
        cap = cv2.VideoCapture(0)
        # 视频分辨率
        resize_w = 1280
        resize_h = 720

        # 控制边距
        frameMargin = 50
        
        # 屏幕尺寸
        screenWidth, screenHeight = pyautogui.size() 

        # 柔和处理参数
        stepX, stepY = 0, 0
        finalX, finalY = 0, 0
        smoothening = 5

        action_trigger_time = {
            'single_click':0,
            'double_click':0,
            'right_click':0
        }


        while cap.isOpened():
            action_zh = ''
            success, self.image = cap.read()
            self.image = cv2.resize(self.image, (resize_w, resize_h))
            if not success:
                print("Camera Failed")
                continue

            # 提高性能
            self.image.flags.writeable = False
            # 转为RGB
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # 镜像，需要根据镜头位置来调整
            self.image = cv2.flip(self.image, 1)
            # 处理手掌
            self.image = handprocess.processOneHand(self.image)

            # 画框框
            cv2.rectangle(self.image, (frameMargin, frameMargin), (resize_w - frameMargin, resize_h - frameMargin),(255, 0, 255), 2)

            # 获取动作
            self.image,action,key_point = handprocess.checkHandAction(self.image,drawKeyFinger=True)

            action_zh = handprocess.action_labels[action]

            if key_point:
                # 映射距离
                x3 = np.interp(key_point[0], (frameMargin, resize_w - frameMargin), (0, screenWidth))
                y3 = np.interp(key_point[1], (frameMargin, resize_h - frameMargin), (0, screenHeight))
                
                # 柔和处理
                finalX = stepX + (x3 - stepX) / smoothening
                finalY = stepY + (y3 - stepY) / smoothening

                now = time.time() 

                if action_zh == '鼠标移动':
                    pyautogui.moveTo(finalX, finalY)
                    # pyautogui.dragTo(finalX,finalY)
                    # 解决移动卡顿的问题
                    # autopy.mouse.move(finalX, finalY)
                    # pyautogui.move()

                elif action_zh == '单击准备':
                    pass
                elif action_zh == '触发单击'  and  (now - action_trigger_time['single_click'] > 2):
                    pyautogui.click()
                    action_trigger_time['single_click'] = now
                    
                elif action_zh == '右击准备':
                    pass
                elif action_zh == '触发右击' and  (now - action_trigger_time['right_click'] > 2):
                    pyautogui.click(button='right')  
                    action_trigger_time['right_click'] = now

                elif action_zh == '向上滑页':
                    pyautogui.scroll(10)
                elif action_zh == '向下滑页':
                    pyautogui.scroll(-10)
                elif action_zh == '鼠标拖拽':
                    pyautogui.dragTo(finalX, finalY,button='left') 
                

                
                stepX, stepY = finalX, finalY


            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            
            
            # 显示刷新率FPS
            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
         
            self.image = utils.cv2AddChineseText(self.image, "帧率: " + str(int(fps_text)),  (10, 30), textColor=(0, 255, 0), textSize=30)
            

            # 显示画面
            self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
            cv2.imshow('virtual mouse', self.image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


if __name__ == '__main__':
    control = VirtualMouse()
    control.recognize()