import cv2
import numpy as np

def imshow_for(title):
    videocapture = cv2.VideoCapture(-1)
    while (True):
        ret, frame = videocapture.read()
        cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    videocapture = cv2.VideoCapture(-1)
    while True:
        ret, frame = videocapture.read()
        cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.ones((len(cap), len(cap[0])))*5
        cap = cap*mask
        cap[cap >= 255] = 255
        cap = cap.astype(np.uint8)
        cv2.imshow(title, cap)
        if cv2.waitKey(1) == 27:
            break

imshow_for("for")
imshow_fun("fun")