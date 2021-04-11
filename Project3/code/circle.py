import cv2
import numpy as np

img_ori = cv2.imread("pic/cst.jpg")
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 119, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("bin", img_binary)

circles = cv2.HoughCircles(img_binary, cv2.HOUGH_GRADIENT, 1, 40, param1=30, param2=30, minRadius=0)
circles_2 = circles[0, :, :]
circles_2 = np.uint16(np.around(circles_2))


for circle in circles_2:
    # draw the outer circle
    cv2.circle(img_ori,(circle[0], circle[1]), circle[2],(0,255,0),2)
    # draw the center of the circle
    # cv.circle(img_ori,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("circle", img_ori)

cv2.waitKey(0)
cv2.destroyAllWindows