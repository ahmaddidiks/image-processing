import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0);

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('Canny');
cv2.namedWindow('Contour');

cv2.createTrackbar('Threshold 1', 'Canny', 0, 300, nothing)
cv2.createTrackbar('Threshold 2', 'Canny', 0, 300, nothing)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while True:
    
    ret, bgr_image = cap.read()

    thresh1 = cv2.getTrackbarPos('Threshold 1', 'Canny')
    thresh2 = cv2.getTrackbarPos('Threshold 2', 'Canny')

    canny = cv2.Canny(bgr_image, thresh1, thresh2)
    
    _, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(bgr_image, contours, -1, (255,15,20), 1) #formatnya BGR

    cv2.imshow('Canny', canny)
    cv2.imshow('Contour', bgr_image)
    
 
    if cv2.waitKey(25) & 0xFF == ord('q'):
          break
 
cap.release()
cv2.destroyAllWindows()
