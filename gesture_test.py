import cv2
import mediapipe as mp
import time
import numpy as np
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        初始化手势检测器对象。
        Args:
            mode (bool): 是否检测多只手。默认为False, 只检测单只手。
            maxHands (int): 最多检测的手的数量。默认为2。
            detectionCon (float): 手势检测的置信度阈值。默认为0.5。
            trackCon (float): 手势跟踪的置信度阈值。默认为0.5。
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        # 创建 Mediapipe Hands 模块和绘制工具对象
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, 
                                        max_num_hands = self.maxHands, 
                                        model_complexity = 1, 
                                        min_detection_confidence = self.detectionCon, 
                                        min_tracking_confidence = self.trackCon)
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        #                                self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, img, draw=True):
        """
        检测手势并在图像上绘制关键点和连接线。
        Args:
            img (numpy.ndarray): 输入图像。
            draw (bool): 是否在图像上绘制标记。默认为True。
        Returns:
            numpy.ndarray: 绘制了关键点和连接线的图像。
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    
if __name__ == '__main__':

    detector = handDetector()
    # For single image
    # hand_img_path = '/Users/2020chanzi/Desktop/code/gesture/test_img/WechatIMG1202.jpg'
    # hand_img_array = cv2.imread(hand_img_path)
    # hand_img_array = np.array(hand_img_array)
    # drawed_img = detector.findHands(hand_img_array)
    # cv2.imshow("test", drawed_img)
    # cv2.waitKey()

    # For video stream
    import numpy as np  #导入科学计算库numpy
    cap = cv2.VideoCapture(0)
    #打开失败
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    #打开成功
    while True:
        #如果正确读取帧，ret为True
        ret, frame = cap.read()
        #读取失败，则退出循环
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #图像处理-转换为灰度图
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #显示画面
        drawed_img = detector.findHands(frame)
        cv2.imshow('frame', drawed_img)
        #获取键盘按下那个键
        key_pressed = cv2.waitKey(60)
        #如果按下esc键，就退出循环
        if key_pressed == 27:
            break
    cap.release()  #释放捕获器
    cv2.destroyAllWindows() #关闭图像窗口