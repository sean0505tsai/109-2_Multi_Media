import cv2
import numpy as np

square_ori = cv2.imread("pic//squares.jpg")
square_gray = cv2.cvtColor(square_ori, cv2.COLOR_BGR2GRAY)
# square_dilate = cv2.dilate(square_gray, np.ones((2, 2)), iterations = 1)
ret, square_bin = cv2.threshold(square_gray, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("square", square_bin)

num_labels4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(square_bin, connectivity = 4, ltype = None)
num_labels8, labels8, stats8, centroids8 = cv2.connectedComponentsWithStats(square_bin, connectivity = 8, ltype = None)

print("4_connectivity : " + str(num_labels4-1))
print("8_connectivity : " + str(num_labels8-1))


cv2.waitKey(0)
cv2.destroyAllWindows()