# first we need is the webcam
import cv2, numpy as np

# change the value of argument by 0 for default webcam, or by webcam ID
cap = cv2.VideoCapture(0)
# ID number three for width
cap.set(3, 640)
# ID number four for height
cap.set(4, 480)
# ID number ten for brightness
cap.set(10, 100)

myColors = [[25,130,215,179,255,255]]

# DEFINE COLOR
def findColor(img, myColors):
    # 1. convert into HSV space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 2. create a mask to filter out our images
    for color in myColors:
        lower = np.array( color[0:3] )
        upper = np.array( color[3:6] )
        mask = cv2.inRange(imgHSV, lower, upper)

        getContour(mask)
        # cv2.imshow(str(color[0]), mask)

def getContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for each contour, we are going to find the area fest
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        # draw the contour to 'imgContour' if area is greater than 500px
        # .drawContours() : image, contours, contourIndex, color, thickness
        if area > 500:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            # length of each contour arc by arc
            peri = cv2.arcLength(cnt, True)
            # get the point of corner point
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # create object corner / bounding boxes
            x, y, w, h = cv2.boundingRect(approx)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    findColor(img, myColors)
    cv2.imshow("result", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break