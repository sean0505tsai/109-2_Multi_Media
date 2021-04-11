import cv2
import numpy as np

pic_ori = cv2.imread("pic/coin.jpg")
pic_gray = cv2.cvtColor(pic_ori, cv2.COLOR_BGR2GRAY)
pic_resize = cv2.resize(pic_gray, (500, 282))
cv2.imshow("resize", pic_resize)


'''-----sobel-----'''
pic_sobel = cv2.Sobel(pic_resize, cv2.CV_16S, 1, 1)
sobel = cv2.convertScaleAbs(pic_sobel)
cv2.imshow("sobel", sobel)


'''-----scharr-----'''
pic_scharrx = cv2.Scharr(pic_resize, cv2.CV_16S, 1, 0)
scharrx = cv2.convertScaleAbs(pic_scharrx)
pic_scharry = cv2.Scharr(pic_resize, cv2.CV_16S, 0, 1)
scharry = cv2.convertScaleAbs(pic_scharry)
scharr = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

cv2.imshow("scharr", scharr)


'''-----laplacian-----'''
pic_laplacian = cv2.Laplacian(pic_resize, cv2.CV_16S, ksize=3)
laplacian = cv2.convertScaleAbs(pic_laplacian)
cv2.imshow("laplacian", laplacian)


'''-----canny-----'''
canny = cv2.Canny(pic_resize, 50, 150)
cv2.imshow("canny", canny)


cv2.waitKey(0)
cv2.destroyAllWindows()