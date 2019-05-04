import cv2
import numpy as np

num_img = cv2.imread("./images/5374.png")
# num_img = cv2.imread("./images/5739.png")
# print(num_img.shape)
# 灰度化
gray_img = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray_img)

# 中值滤波
blur_img = cv2.medianBlur(gray_img, 3)
cv2.imshow("Blur", blur_img)

max = float(gray_img.max())
min = float(gray_img.min())
x = max - ((max - min) / 2)
# 二值化处理
ret, binary_img = cv2.threshold(gray_img, x, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary_img)

# "?"-形结构元素
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
# 膨胀
dilate_img = cv2.dilate(binary_img, kernel_dilate)
cv2.imshow("Dilate", dilate_img)
# erode_img =dilate_img

kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
# 腐蚀
erode_img = cv2.erode(dilate_img, kernel_erode)
cv2.imshow("Erode", erode_img)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # 开运算，闭运算，形态梯度，顶帽，黑帽
# open = cv2.morphologyEx(binary_img,cv2.MORPH_OPEN,kernel)
# cv2.imshow("Open", open)
# close = cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel)
# cv2.imshow("Close", close)
# gradient = cv2.morphologyEx(binary_img,cv2.MORPH_GRADIENT,kernel)
# cv2.imshow("Gradient", gradient)
# tophat = cv2.morphologyEx(binary_img,cv2.MORPH_TOPHAT,kernel)
# cv2.imshow("Tophat", tophat)
# blackhat = cv2.morphologyEx(binary_img,cv2.MORPH_BLACKHAT,kernel)
# cv2.imshow("Blackhat", blackhat)

# 寻找图像轮廓
# contours, hierarchy = cv2.findContours(erode_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# for i in range(len(contours)):
#     contour_img = cv2.drawContours(num_img, contours[i], -1, (0, 0, 255))
#     cv2.imshow("Contour" + str(i), contour_img)
#
#     # 最小矩形
#     x, y, w, h = cv2.boundingRect(contours[i])
#     # 最小斜矩形
#     # rect = cv2.minAreaRect(contours[i])
#     # 画出矩形
#     bound = cv2.rectangle(erode_img.copy(), (x, y), (x + w, y + h), (0, 0, 255))
#     cv2.imshow("bound" + str(i), bound)

# # 边缘检测
# soble = cv2.Sobel(binary_img, cv2.CV_16S, 1, 1, ksize=3)
# abs_soble = cv2.convertScaleAbs(soble)
# cv2.imshow("Sobel", abs_soble)
#
# canny = cv2.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])
# cv2.imshow("Canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
