from itertools import groupby

import cv2
import numpy as np
from PIL import Image


# def vertical(img):
#     """传入二值化后的图片进行垂直投影"""
#     print(img.shape)
#     pixdata = img
#     row,column = img.shape[0],img.shape[1]
#     result = []
#     for y in range(column):
#         black = 0
#         for x in range(row):
#             if pixdata[x,y] == 0:
#                 black += 1
#         result.append(black)
#     return result


def get_nearby_pix_value(img_pix, row, column, j):
    """获取临近5个点像素数据
       n5   n0   n4
       n1   n2   n3"""
    if j == 1:
        return 0 if img_pix[row + 1, column - 1] == 0 else 1
    elif j == 2:
        return 0 if img_pix[row + 1, column] == 0 else 1
    elif j == 3:
        return 0 if img_pix[row + 1, column + 1] == 0 else 1
    elif j == 4:
        return 0 if img_pix[row, column + 1] == 0 else 1
    elif j == 5:
        return 0 if img_pix[row, column - 1] == 0 else 1
    else:
        raise Exception("get_nearby_pix_value error")


def get_route(img, start_column):
    """获取滴水路径"""
    # print(img[100])
    column_limit = img.shape[1] - 1
    row_limit = img.shape[0] - 1
    end_route = []
    cur_p = (0, start_column)
    last_p = cur_p
    end_route.append(cur_p)

    a = 0
    while cur_p[0] < row_limit:
        sum_n = 0
        max_w = 0
        next_row = cur_p[0]
        next_column = cur_p[1]

        for i in range(1, 6):
            cur_w = get_nearby_pix_value(img, cur_p[0], cur_p[1], i) * (6 - i)
            sum_n += cur_w
            if max_w < cur_w:
                max_w = cur_w

        if sum_n == 0 or sum_n == 15:
            max_w = 4

        if max_w == 1:
            next_row = cur_p[0]
            next_column = cur_p[1] - 1
        elif max_w == 2:
            next_row = cur_p[0]
            next_column = cur_p[1] + 1
        elif max_w == 3:
            next_row = cur_p[0] + 1
            next_column = cur_p[1] + 1
        elif max_w == 5:
            next_row = cur_p[0] + 1
            next_column = cur_p[1] - 1
        elif max_w == 6:
            next_row = cur_p[0] + 1
            next_column = cur_p[1]
        elif max_w == 4:
            if next_column > last_p[1]:
                # 向右
                next_row = cur_p[0] + 1
                next_column = cur_p[1] + 1
            else:
                next_row = cur_p[0] + 1
                next_column = cur_p[1]
        else:
            raise Exception("get end route error")

        if next_column > column_limit:
            next_column = column_limit
        if next_column < 0:
            next_column = 0

        cur_p = (next_row, next_column)
        end_route.append(cur_p)

        # 前一个位置，用于判断是否有向右的惯性
        a += 1
        if a == 3:
            last_p = cur_p
            a = 0

    return end_route


def do_split(source_image, starts, filter_ends):
    """
    具体实行切割
    : param starts: 每一行的起始点 tuple of list
    : param ends: 每一行的终止点
    """

    left = starts[0][0]
    top = starts[0][1]
    right = filter_ends[0][0]
    bottom = filter_ends[0][1]
    for i in range(len(starts)):
        left = min(starts[i][0], left)
        top = min(starts[i][1], top)
    for i in range(len(filter_ends)):
        right = max(filter_ends[i][0], right)
        bottom = max(filter_ends[i][1], bottom)
    row = right - left + 1
    column = bottom - top + 1

    print(row)
    print(column)
    img = np.zeros((row, column, 3), np.uint8)
    # 使用白色填充图片区域,默认为黑色
    img.fill(255)

    for i in range(row):
        start = starts[i]
        end = filter_ends[i]
        for y in range(start[1], end[1] + 1):
            if source_image[start[0], y] == 0:
                img[start[0] - left][y - top] = 0


    return img


def drop_fall(img, start_column):
    """滴水分割"""
    row, column = img.shape[0], img.shape[1]

    # 4 开始滴水算法
    start_route = []
    for x in range(row):
        start_route.append((x, 0))

    route = get_route(img, start_column)
    print(start_route)
    print(route)

    img1 = do_split(img, start_route, route)
    cv2.imwrite('./images/Drop_Fall/cuts-d-1.png', img1)

    start_route = route
    route = []
    for x in range(row):
        route.append((x, column - 1))
    print(start_route)
    print(route)
    img2 = do_split(img, start_route, route)
    cv2.imwrite('./images/Drop_Fall/cuts-d-2.png', img2)


if __name__ == '__main__':
    num_img = cv2.imread("./images/5374.png")

    # 灰度化
    gray_img = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    blur_img = cv2.medianBlur(gray_img, 3)

    max_value = float(gray_img.max())
    min_value = float(gray_img.min())
    x = max_value - ((max_value - min_value) / 2)
    # 二值化处理
    ret, binary_img = cv2.threshold(gray_img, x, 255, cv2.THRESH_BINARY)

    # "?"-形结构元素
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # 膨胀
    dilate_img = cv2.dilate(binary_img, kernel_dilate)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # 腐蚀
    erode_img = cv2.erode(dilate_img, kernel_erode)

    # img2 = cv2.flip(erode_img, -1)
    # img = np.rot90(img2)
    drop_fall(erode_img, 140)
