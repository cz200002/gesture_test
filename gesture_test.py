import cv2
import mediapipe as mp
import time
import numpy as np
# from HandTrackingModule import HandDetector
 
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        mode (bool): 是否检测多只手。默认为False, 只检测单只手。
        maxHands (int): 最多检测的手的数量。默认为2。
        detectionCon (float): 手势检测的置信度阈值。默认为0.5。
        trackCon (float): 手势跟踪的置信度阈值。默认为0.5。
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, 
                                        max_num_hands = self.maxHands, 
                                        model_complexity = 1, 
                                        min_detection_confidence = self.detectionCon, 
                                        min_tracking_confidence = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, img, draw=True):
        """
        Input:
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


# -*- coding:utf-8 -*-
 
"""
@ By: ZhengXuan
@ Date: 2024-4-21
"""
 
import cv2
import mediapipe as mp
import numpy as np
 
 
class HandDetector:
    """
    使用mediapipe库查找手。导出地标像素格式。添加了额外的功能。
    如查找方式，许多手指向上或两个手指之间的距离。而且提供找到的手的边界框信息。
    """
 
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: 在静态模式下，对每个图像进行检测
        :param maxHands: 要检测的最大手数
        :param detectionCon: 最小检测置信度
        :param minTrackCon: 最小跟踪置信度
        """
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = False
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
 
        # 初始化手部识别模型
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils  # 初始化绘图器
        self.tipIds = [4, 8, 12, 16, 20]  # 指尖列表
        self.fingers = []  # 存储手的状态
        self.lmList = []  # 储检测到的手部的每个关键点的坐标
 
    def findHands(self, img, draw=True):
        """
        从图像(BRG)中找到手部。
        :param img: 用于查找手的图像。
        :param draw: 在图像上绘制输出的标志。
        :return: 带或不带图形的图像
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将传入的图像由BGR模式转标准的Opencv模式——RGB模式，
        self.results = self.hands.process(imgRGB)  # 处理图像，返回包含检测到的手部信息的结果。这个结果通常包含了手部的关键点坐标
 
        # 画出手的关键点和线条
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw=True):
        """
        查找单手的地标并将其放入列表中像素格式。还可以返回手部周围的边界框。
        :param img: 要查找的主图像
        :param handNo: 如果检测到多只手，则为手部id
        :param draw: 在图像上绘制输出的标志。(默认绘制矩形框)
        :return: 像素格式的手部关节位置列表；手部边界框
        """
        # 保存关键点的像素坐标
        xList = []
        yList = []
        bbox = []
        bboxInfo = []  # 保存首部检测框信息
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # 遍历手部关键点，id表示关键点下标，lm表示关键点对象
                h, w, c = img.shape
                #  lm是存储的是关键点归一化（0~1）的相对位置，
                px, py = int(lm.x * w), int(lm.y * h)  # 转换为图像中的像素坐标
                xList.append(px)
                yList.append(py)
                self.lmList.append([px, py])
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)  # 用红点标记关键点
            # 获取手关键点的左上点和右下点
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            # 边界框信息存储
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            # 边界框中心坐标
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + (bbox[3] // 2)
            # id含义是指的手部最后一个关键点的下标
            bboxInfo = {"id": id, "bbox": bbox, "center": (cx, cy)}
 
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)
 
        return self.lmList, bboxInfo
 
    def fingcurved(self):
        """
        查找除拇指外的四个手指弯曲状态
        计算方式：
            取出除了大拇指以外的四个手指指尖坐标a1、a2、a3、a4（对应地标8，12，16，20），
            然后取出地标为6，10，14，18的坐标b1、b2、b3、b4（即每个指尖以下第二个关节），
            通过比较指尖（a1、a2、a3、a4）到手腕地标（0）和指关节(b1、b2、b3、b4)到地标0的欧几里得距离，
            即可区分手指是否弯曲
        :return: 弯曲手指的列表
        """
        finger = []
        for id in range(1, 5):
            point1 = np.array((self.lmList[self.tipIds[id]][0], self.lmList[self.tipIds[id]][1]))
            point2 = np.array((self.lmList[self.tipIds[id] - 2][0], self.lmList[self.tipIds[id] - 2][1]))
            point3 = np.array((self.lmList[0][0], self.lmList[0][1]))
            if np.linalg.norm(point1 - point3) < np.linalg.norm(point2 - point3):  # 计算两点之间的距离
                finger.append(1)
            else:
                finger.append(0)
 
        return finger
 
    def okgesture(self):
        """
        特殊手势处理：判断是否手势为ok
        判断方式：
            ok手势，其拇指指尖地标a0和食指指尖地标a1十分接近，于是我们这样处理：如果中指、无名  
            指、小拇指伸直并且食指指尖到大拇指指尖的距离小于食指指尖到中指指尖距离则断定为ok手   
            势。
        """
        f1, f2, f3, f4 = self.fingcurved()
        if (f2 == 0 and f3 == 0 and f4 == 0):
            point1 = np.array((self.lmList[8][0], self.lmList[8][1]))
            point2 = np.array((self.lmList[4][0], self.lmList[4][1]))
            point3 = np.array((self.lmList[12][0], self.lmList[12][1]))
            if np.linalg.norm(point1 - point2) < np.linalg.norm(point1 - point3):
                return True
 
    def handType(self):
        """
        检查传入的手部是左还是右
        ：return: "Right" 或 "Left"
        """
        if self.results.multi_hand_landmarks:
            if self.lmList[17][0] < self.lmList[5][0]:
                return "Right"
            else:
                return "Left"


class Main:
 
    def __init__(self):
        self.detector = None
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 以视频流传入
        self.camera.set(3, 1280)  # 设置分辨率
        self.camera.set(4, 720)
 
    def Gesture_recognition(self):
        self.detector = HandDetector()
        gesture_buffer = [None] * 3  # 只有连续三帧都为同一手势，才输出该手势，提高识别鲁棒性
        while True:
            ret, img = self.camera.read()
            if ret:
                img = self.detector.findHands(img)  # 获取你的手部的关键点信息
                cv2.imshow('hand',img)
                lmList, bbox = self.detector.findPosition(img)  # 获取你手部的关键点的像素坐标和边界框
                if lmList:
                    x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                    f1, f2, f3, f4 = self.detector.fingcurved()
                    # 根据手指弯曲状态识别手势并在图像上显示相应文本
                    if f1 == 0 and f2 == 1 and f3 == 1 and f4 == 0:
                        gesture = "twist"
                    elif f1 == 0 and f2 == 1 and f3 == 1 and f4 == 1:
                        gesture = "forward"
                    elif self.detector.okgesture():
                        gesture = "right_move"
                    elif f1 == 0 and f2 == 0 and f3 == 0 and f4 == 0:
                        gesture = "back"
                    elif f1 == 0 and f2 == 0 and f3 == 1 and f4 == 1:
                        gesture = "left_move"
                    elif f1 == 1 and f2 == 1 and f3 == 1 and f4 == 0:
                        gesture = "left_twist"
                    elif f1 == 0 and f2 == 1 and f3 == 0 and f4 == 0:
                        gesture = "right_twist"
                    else:
                        gesture = None
                    if gesture:
                        cv2.putText(img, gesture, (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                        gesture_buffer.insert(0, gesture)
                        gesture_buffer.pop()
 
                cv2.imshow("camera", img)
                if gesture_buffer[0] is not None and all(ges == gesture_buffer[0] for ges in gesture_buffer):
                    self.print_gesture(gesture_buffer[0])
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            # 通过关闭按钮退出程序
            cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break # 按下q退出
 
    def print_gesture(self, gesture):
        if gesture == "twist":
            print("原地扭身")
        elif gesture == "forward":
            print("前进")
        elif gesture == "right_move":
            print("右平移")
        elif gesture == "back":
            print("后退")
        elif gesture == "left_twist":
            print("左旋转")
        elif gesture == "right_twist":
            print("右旋转")
        elif gesture == "left_move":
            print("左平移")
 
 
if __name__ == '__main__':

    # detector = handDetector()
    # For single image
    # hand_img_path = '/Users/2020chanzi/Desktop/code/gesture/test_img/WechatIMG1202.jpg'
    # hand_img_array = cv2.imread(hand_img_path)
    # hand_img_array = np.array(hand_img_array)
    # drawed_img = detector.findHands(hand_img_array)
    # cv2.imshow("test", drawed_img)
    # cv2.waitKey()

    
    # For video stream
    # import time
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
        
    # while True:
    #     start_time = time.time()
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     pause_time = time.time()
    #     drawed_img = detector.findHands(frame)
    #     cv2.imshow('frame', drawed_img)
    #     end_time = time.time()
    #     print('-'*10)
    #     print('cal time', end_time - pause_time)
    #     print('all time', end_time - start_time) # about 50Hz
    #     key_pressed = cv2.waitKey(60)
    #     if key_pressed == 27:
    #         break
    # cap.release()
    # cv2.destroyAllWindows() 
    

    main = Main()
    main.Gesture_recognition()