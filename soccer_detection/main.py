import cv2
import numpy as np


def ground(r, g, b):
    if g > r > b:
        return 0
    else:
        return 255


def img_sum(a, b):
    if a + b > 255:
        return 255
    else:
        return a + b


vground = np.vectorize(ground)
vimg_sum = np.vectorize(img_sum)
video = cv2.VideoCapture("soccer.mp4")

while True:
    ret, frame = video.read()

    b, g, r = cv2.split(frame)

    new_frame = vground(r, g, b).astype(np.uint8)
    cv2.imshow("ground", new_frame)
    cv2.imshow("original", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=5)
    cv2.imshow("sobel5x5", sobel)

    cv2.imshow("sobel+ground", vimg_sum(sobel, new_frame))

    if cv2.waitKey(1) == 27 or not ret:
        break
