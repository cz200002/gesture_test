import cv2
import mediapipe as mp
import time
import numpy as np
 
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
    import time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        pause_time = time.time()
        drawed_img = detector.findHands(frame)
        cv2.imshow('frame', drawed_img)
        end_time = time.time()
        print('-'*10)
        print('cal time', end_time - pause_time)
        print('all time', end_time - start_time) # about 50Hz
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break
    cap.release()
    cv2.destroyAllWindows() 
    