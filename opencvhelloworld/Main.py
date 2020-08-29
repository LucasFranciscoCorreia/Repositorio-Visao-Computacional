import cv2

image = cv2.imread("Lenna.png")

cv2.imshow("image", image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray image", gray_image)

cv2.imwrite("Lenna_gray.png", gray_image)

cv2.waitKey()
