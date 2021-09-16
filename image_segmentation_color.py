import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/04/10/19/pinyon-jay-bird.jpg

img = cv.imread('D:\\image_segmentation\\pinyon-jay-bird.jpg')
img = cv.resize(img, (960, 540))

low_blue = np.array([0, 0, 0])
high_blue = np.array([255, 0, 0])


def set_low_blue(l_blue):
    print("low_blue = ", l_blue)
    global low_blue
    low_blue = np.array([l_blue, 0, 0])
    # 阈值分割：确定要提取的所有像素的阈值
    mask = cv.inRange(hsv, low_blue, high_blue)
    # 显示由Mask作为边界的图像
    res = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("image_segmentation", res)


def set_high_blue(h_blue):
    print("high_blue = ", h_blue)
    global high_blue
    high_blue = np.array([h_blue, 255, 255])
    mask = cv.inRange(hsv, low_blue, high_blue)
    # 显示由Mask作为边界的图像
    res = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("image_segmentation", res)


# 对图像进行模糊操作，以减少图像中的细微差异
blur = cv.blur(img, (5, 5))
blur0 = cv.medianBlur(blur, 5)
blur1 = cv.GaussianBlur(blur0, (5, 5), 0)
blur2 = cv.bilateralFilter(blur1, 9, 75, 75)

# 需要将图像从BGR（蓝绿色红色）转换为HSV（色相饱和度值）。为什么我们要从BGR空间中转到HSV空间中？因为像素B，G和R的取值与落在物体上的光相关，因此这些值也彼此相关，无法准确描述像素。相反，HSV空间中，三者相对独立，可以准确描述像素的亮度，饱和度和色度。
hsv = cv.cvtColor(blur2, cv.COLOR_BGR2HSV)

cv.namedWindow('image_segmentation')
cv.createTrackbar("low_blue", 'image_segmentation', 0, 55, set_low_blue)
cv.createTrackbar("high_blue", 'image_segmentation', 255, 255, set_high_blue)

while 1:
    if cv.waitKey(1) == ord('q'):
        break
    l_blue = cv.getTrackbarPos('low_blue', 'image_segmentation')
    h_blue = cv.getTrackbarPos('high_blue', 'image_segmentation')
cv.destroyAllWindows()


# cv2.createTrackbar(trackbarName, windowName, value, count, onChange)
# cv2.getTrackbarPos(trackbarname, winname)