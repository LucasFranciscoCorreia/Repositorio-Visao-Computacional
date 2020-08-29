import cv2

videocapture = cv2.VideoCapture(0)

while(True):
    capture = videocapture.read()
    cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera", capture)
    if(cv2.waitKey(1) == 27):
        break