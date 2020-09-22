import cv2
import numpy as np


def trackChaned(x):
    pass


def skin_extract(y, cr, cb):
    value = cv2.getTrackbarPos("skin_threshold", "Constants")
    # value = 0
    y[y < 54] = value
    y[y > 163] = value
    cr[cr < 131] = value
    cr[cr > 157] = value
    cb[cb < 110] = value
    cb[cb > 135] = value
    y[cb < 77] = cb[cb < 77] = cr[cb < 77] = value
    y[cb > 127] = cb[cb > 127] = cr[cb > 127] = value

    y[cr < 133] = cb[cr < 133] = cr[cr < 133] = value
    y[cr > 173] = cb[cr > 173] = cr[cr > 173] = value

    return y, cr, cb


def convert_YCrCb(frame):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return result


def background_subtraction(frame):
    result = frame -background
    return result


def threshold_binarization_morphology(y, cr, cb):
    _, y = cv2.threshold(y, cv2.getTrackbarPos("ymin", "Constants"), 255, cv2.THRESH_BINARY)
    _, cr = cv2.threshold(cr, cv2.getTrackbarPos("crmin", "Constants"), 255, cv2.THRESH_BINARY)
    _, cb = cv2.threshold(cb, cv2.getTrackbarPos("cbmin", "Constants"), 255, cv2.THRESH_BINARY)

    return y, cr, cb


def morphology(frame, morph):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(frame, morph, kernel)


def canny_edge(frame):
    return cv2.Canny(frame, cv2.getTrackbarPos("cannyMin", "Constants"), cv2.getTrackbarPos("cannyMax", "Constants"))


def first_module(frame):
    ycrcb = convert_YCrCb(frame)

    ycrcb_s = background_subtraction(ycrcb)

    y, cr, cb = cv2.split(ycrcb_s)
    y, cr, cb = threshold_binarization_morphology(y, cr, cb)

    mask = morphology(y, cv2.MORPH_OPEN)

    ycrcb_and = cv2.bitwise_and(ycrcb, cv2.merge((mask, mask, mask)))
    y, cr, cb = cv2.split(ycrcb_and)
    cv2.imshow("mask", mask)
    cv2.imshow("y1", y)
    y, cr, cb = skin_extract(y, cr, cb)
    cv2.imshow("y2", y)

    gaussian = cv2.GaussianBlur(y, (3, 3), 25)

    # cv2.imshow("and", ycrcb_and)
    # cv2.imshow("rgb", rgb)
    # cv2.imshow("and", ycrcb_and)
    # cv2.imshow("canny", canny)
    cv2.imshow("gaussian", gaussian)
    # cv2.imshow("borders", mask - canny)
    # cv2.imshow("y", y)
    return gaussian


cam = cv2.VideoCapture("http://192.168.15.3:8080/video")
ret, background = cam.read()
cv2.imwrite("background.jpg", background)
# background = cv2.imread("background.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2YCrCb)
ret, frame = cam.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    # frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    first_module_result = first_module(frame)
    contours, hierarchy = cv2.findContours(first_module_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("mask", first_module_result)
    # cv2.imshow("original", frame)
    # cv2.imshow("and", np.bitwise_and(cv2.merge((first_module_result,first_module_result,first_module_result)), frame))
    # cv2.imshow("finding contours", first_module_result)
    # cv2.drawContours(first_module_result, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("contours", first_module_result)
    # cv2.imshow("contours", np.bitwise_and(frame, cv2.merge((first_module_result,first_module_result,first_module_result))))
    if cv2.waitKey(1) == 27:
        break
