import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.namedWindow("Photo_Detect")  # 定义一个窗口
cap = cv2.VideoCapture(0)  # 捕获摄像头图像  0位默认的摄像头 笔记本的自带摄像头  1为外界摄像头
while True:                  # 值为1不断读取图像
    ret, frame = cap.read()  # 视频捕获帧
    # cv2.imwrite('D:\\image_segmentation\\cap_RGB.jpg', frame)     # 写入捕获到的视频帧  命名为cap_RGB.jpg
    cv2.imshow('Photo_Detect', frame)     # 显示窗口 查看实时图像
    # 按S读取灰度图
    if (cv2.waitKey(1) & 0xFF) == ord('S'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
        # gray = cv2.resize(gray, (640, 480))  # 图像大小为640*480
        # cv2.imshow('cap', gray)               # 显示灰度图窗口
        cv2.imwrite('D:\\image_segmentation\\cap_RGB.jpg', frame)
        cv2.imwrite('D:\\image_segmentation\\cap.jpg', gray)          # 写入灰度图

    if cv2.waitKey(1) & 0xFF == ord('Q'):   # 按Q关闭所有窗口  一次没反应的话就多按几下
        break
#执行完后释放窗口
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
