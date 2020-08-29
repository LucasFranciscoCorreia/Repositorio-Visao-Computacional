import cv2

videocapture = cv2.VideoCapture(-1)

while(True):
    ret, frame =  videocapture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera", frame)
    if(cv2.waitKey(1) == 27):
        break