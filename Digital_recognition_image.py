import cv2
from matplotlib import pylab
import matplotlib.image as image

# num_img = image.imread("./images/5374.png")
# print(num_img.shape)
# r_channel = num_img[:, :, 0]
# r_channel[r_channel == 1] = 0
# print(num_img.shape)
# pylab.gray()
# pylab.imshow(r_channel)
# pylab.show()

ima = cv2.imread('./result/82.jpg')
res = cv2.resize(ima, (28, 28), interpolation=cv2.INTER_CUBIC)
cv2.imshow('iker', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
