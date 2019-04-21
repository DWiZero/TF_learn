import cv2
import numpy as np
from PIL import Image
import os.path
from skimage import io,data
def Binarization01(img):
    '''二值化处理函数'''
    max = float(img.max())
    min = float(img.min())

    x = max - ((max - min)/ 2)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh;阈值函数
    ret, thresh = cv2.threshold(img, x, maxval=255, type=cv2.THRESH_BINARY)
    print(ret)
    # 返回二值化后的黑白图像
    return thresh

def Binarization02(img):
    '''滤波+二值化'''
    #变为灰度图像
    # gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img= img
    #均值滤波  去除噪声
    # kernel=np.ones((3,3),np.float32)/9
    # gray_img=cv2.filter2D(gray_img,-1,kernel)

    # 中值滤波
    gray_img = cv2.medianBlur(gray_img,3)

    max = float(img.max())
    min = float(img.min())

    x = max - ((max - min) / 2)
    #二值化处理
    ret,thresh=cv2.threshold(gray_img,x,255,cv2.THRESH_BINARY)

    return thresh

def find_end(start,arg,black,white,width,black_max,white_max):
    end=start+1
    for m in range(start+1,width-1):
        if (black[m] if arg else white[m])>(0.98*black_max if arg else 0.98*white_max):
            end=m
            break
    return end

num_img = cv2.imread("./images/56528.jpg")
# num_img = cv2.imread("./images/5374.png")
# num_img = cv2.imread("./images/5739.png")
# print(num_img.shape)
gray_img = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)

# binary_img = gray_img
# binary_img = Binarization01(gray_img)
binary_img = Binarization02(gray_img)
cv2.imshow("binary", binary_img)
canny = binary_img

soble = cv2.Sobel(binary_img, cv2.CV_16S, 1, 1, ksize=3)
abs_soble = cv2.convertScaleAbs(soble)
cv2.imshow("Sobel", abs_soble)

edge = cv2.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])
cv2.imshow("edge", canny)

thresh = Binarization01(canny)
# 记录黑白像素总和
white = []
black = []
height = thresh.shape[0]  # 263
width = thresh.shape[1]  # 400
# print('height',height)
# print('width',width)
white_max = 0
black_max = 0
# 计算每一列的黑白像素总和
for i in range(width):
    line_white = 0
    line_black = 0
    for j in range(height):
        if thresh[j][i] == 255:
            line_white += 1
        if thresh[j][i] == 0:
            line_black += 1
    white_max = max(white_max, line_white)
    black_max = max(black_max, line_black)
    white.append(line_white)
    black.append(line_black)
    # print('white', white)
    # print('black', black)
# arg为true表示黑底白字，False为白底黑字
arg = True
if black_max < white_max:
    arg = False

n = 1
start = 1
end = 2
s_width = 28
s_height = 28
while n < width - 2:
    n += 1
    # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
    if (white[n] if arg else black[n]) > (0.02 * white_max if arg else 0.02 * black_max):
        start = n
        end = find_end(start, arg, black, white, width, black_max, white_max)
        n = end
        if end - start > 5:
            cj = thresh[1:height, start:end]

            # new_image = cj.resize((s_width,s_height),Image.BILINEAR)
            # cj=cj.reshape(28, 28)
            print("result/%s.jpg" % (n))
            # 保存分割的图片 by cayden
            # cj.save("result/%s.jpg" % (n))
            infile = "result/%s.jpg" % (n)
            io.imsave(infile, cj)

            # im = Image.open(infile)
            # out=im.resize((s_width,s_height),Image.BILINEAR)
            # out.save(infile)

            cv2.imshow('cutlicense', cj)
            cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
