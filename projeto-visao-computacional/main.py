import math

import cv2
import numpy as np


def trackChaned(x):
    pass


def skin_extract(y, cr, cb):
    # value = cv2.getTrackbarPos("skin_threshold", "Constants")
    value = 0

    cbmin = 60
    cbmax = 127
    y[cb < cbmin] = cb[cb < cbmin] = cr[cb < cbmin] = value
    y[cb > cbmax] = cb[cb > cbmax] = cr[cb > cbmax] = value

    crmin = 133
    crmax = 180
    y[cr < crmin] = cb[cr < crmin] = cr[cr < crmin] = value
    y[cr > crmax] = cb[cr > crmax] = cr[cr > crmax] = value
    y[y != value] = 255
    cr[cr != value] = 255
    cb[cb != value] = 255
    return y, cr, cb


def convert_YCrCb(frame):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return result


def background_subtraction(frame):
    background = cv2.imread("background.jpg")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2YCrCb)

    result = frame - background
    return result


def threshold_binarization_morphology(y, cr, cb):
    _, y = cv2.threshold(y, cv2.getTrackbarPos("ymin", "Constants"), 255, cv2.THRESH_BINARY)
    _, cr = cv2.threshold(cr, cv2.getTrackbarPos("crmin", "Constants"), 255, cv2.THRESH_BINARY)
    _, cb = cv2.threshold(cb, cv2.getTrackbarPos("cbmin", "Constants"), 255, cv2.THRESH_BINARY)

    return y, cr, cb


def morphology(frame, morph):
    kernel = np.ones((9, 9), np.uint8)
    return cv2.morphologyEx(frame, morph, kernel)


def canny_edge(frame):
    return cv2.Canny(frame, cv2.getTrackbarPos("cannyMin", "Constants"), cv2.getTrackbarPos("cannyMax", "Constants"))


def first_module(frame):
    ycrcb = convert_YCrCb(frame)

    ycrcb_s = background_subtraction(ycrcb)

    y, cr, cb = cv2.split(ycrcb_s)

    y, cr, cb = threshold_binarization_morphology(y, cr, cb)

    mask = cv2.merge((y, y, y))

    mask = morphology(mask, cv2.MORPH_OPEN)

    ycrcb_and = cv2.bitwise_and(ycrcb, mask)

    y, cr, cb = cv2.split(ycrcb_and)

    y, cr, cb = skin_extract(y, cr, cb)

    y[y != 0] = 255
    cr[cr != 0] = 255
    cb[cb != 0] = 255

    gaussian = cv2.GaussianBlur(np.bitwise_or(y, cr, cb), (3, 3), 25)

    return gaussian


def second_module(img):
    img = morphology(img, cv2.MORPH_OPEN)

    img_backup = img.copy()

    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    radius = np.amax(dist)

    maxPos = cv2.minMaxLoc(dist)

    pos = maxPos[3]

    roi = int(radius * 3.5)

    aux = cv2.merge((img, img, img))
    cv2.circle(aux, pos, roi, (255, 0, 0), 6)

    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, pos, roi, (255), -1, 8, 0)

    result = np.bitwise_and(img, mask)
    result = np.bitwise_and(result, img_backup)

    contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hullList = []
    defects = []
    aux = cv2.merge((result, result, result))
    for contour in contours:
        cv2.drawContours(aux, [contour], -1, (0, 0, 255), 6)

        hull = cv2.convexHull(contour)
        hullList.append(hull)

        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defect = cv2.convexityDefects(contour, hull)
            if (defect != None):
                defects.append(defect)
        except:
            pass

    if len(contours) > 0 and len(defects) > 0:
        defects = defects
        cnt = contours[0]

        for i in range(defects[0].shape[0]):
            s, e, f, d = defects[0][i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(aux, start, end, [0, 255, 0], 6)
            cv2.circle(aux, far, 6, [255, 0, 0], -1)

    try:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull = cv2.convexHull(approx, returnPoints=False)

        defects = cv2.convexityDefects(approx, hull)

        hull = cv2.convexHull(cnt)

        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        arearatio = ((areahull - areacnt) / areacnt) * 100
        l = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            d = (2 * ar) / a

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        font = cv2.FONT_HERSHEY_SIMPLEX

        if l == 1:
            if arearatio < 22:
                cv2.putText(aux, '0', (0, 50), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(aux, '1', (0, 50), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        elif l == 2:
            cv2.putText(aux, '2', (0, 50), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        elif l == 3:
            cv2.putText(aux, '3', (0, 50), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        elif l == 4:
            cv2.putText(aux, '4', (0, 50), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        elif l == 5:
            cv2.putText(aux, '5', (0, 50), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
    except:
        pass

    return aux


cam = cv2.VideoCapture("http://192.168.15.2:8080/video")
ret, background = cam.read()
cv2.imwrite("background.jpg", background)

cv2.namedWindow('Constants')

cv2.createTrackbar("ymin", "Constants", 0, 255, trackChaned)
cv2.setTrackbarPos("ymin", "Constants", 125)

cv2.createTrackbar("crmin", "Constants", 0, 255, trackChaned)
cv2.setTrackbarPos("crmin", "Constants", 93)

cv2.createTrackbar("cbmin", "Constants", 0, 255, trackChaned)
cv2.setTrackbarPos("cbmin", "Constants", 40)

cv2.createTrackbar("cannyMin", "Constants", 0, 1000, trackChaned)
cv2.setTrackbarPos("cannyMin", "Constants", 30)

cv2.createTrackbar("cannyMax", "Constants", 0, 1000, trackChaned)
cv2.setTrackbarPos("cannyMax", "Constants", 60)

cv2.createTrackbar("skin_threshold", "Constants", 0, 255, trackChaned)

while True:
    ret, frame = cam.read()

    first_module_result = first_module(frame)
    # cv2.imshow("first_module_result", first_module_result)

    second_module_result = second_module(first_module_result)
    cv2.imshow("second_module_result", second_module_result)

    if cv2.waitKey(1) == 27:
        break
