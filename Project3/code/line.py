import cv2
import numpy as np

img_ori = cv2.imread("pic/cst.jpg")
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.bilateralFilter(img_gray, 50, 15, 25)
# cv2.imshow("blur", img_blur)
# img_canny = cv2.Canny(img_blur, 50, 150)
# cv2.imshow("canny", img_canny)

img_gray = cv2.dilate(img_gray, np.ones((2, 2)), iterations=3)      #膨脹
ret, img_binary = cv2.threshold(img_gray, 119, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("floor_dilate", floor_binary)
# img_binary = cv2.erode(floor_binary, np.ones((2, 1)), iterations=1)       #侵蝕回去
# cv2.imshow("floor_erode", floor_binary)

cv2.imshow("bin", img_binary)

lines = cv2.HoughLinesP(img_binary, 1, np.pi/1, 10, 2000, 10)
                    #src, 距離解析度, 角度解析度, 門檻值, 線段最短距離, 最大間隔

for line in lines:

    x1, y1, x2, y2 = line[0]
    cv2.line(img_ori, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("pic", img_ori)

cv2.waitKey(0)
cv2.destroyAllWindows