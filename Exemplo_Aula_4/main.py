import cv2
import numpy as np

def imshow_for(title):
    while (True):
        ret, frame = videocapture.read()
        cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", cap)
        for i in range(len(cap)):
            for j in range(len(cap[i])):
                if cap[i][j] * 5 > 255:
                    cap[i][j] = 255
                else:
                    cap[i][j] = cap[i][j] * 5

        cv2.imshow(title, cap)
        if cv2.waitKey(1) == 27:
            break


def imshow_fun(title):
    while True:
        ret, frame = videocapture.read()
        cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.ones((len(cap), len(cap[0])))*5
        cap2 = cap*mask
        cap2[cap2 >= 255] = 255
        cap2 = cap2.astype(np.uint8)
        cv2.imshow(title, cap2)
        cv2.imshow("original", cap)
        if cv2.waitKey(1) == 27:
            break

videocapture = cv2.VideoCapture(-1)

imshow_for("for")
# imshow_fun("fun")