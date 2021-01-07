import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0);

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('Frame');
cv2.namedWindow('Hasil');

cv2.createTrackbar('Threshold 1', 'Hasil', 0, 300, nothing)
cv2.createTrackbar('Threshold 2', 'Hasil', 0, 300, nothing)

while True:
    
    ret, bgr_image = cap.read()

    thresh1 = cv2.getTrackbarPos('Threshold 1', 'Hasil')
    thresh2 = cv2.getTrackbarPos('Threshold 2', 'Hasil')

    # deteksi tepi
    canny = cv2.Canny(bgr_image, thresh1, thresh2)
    
    cv2.imshow('Frame', bgr_image)
    cv2.imshow('Hasil', canny)
 
    if cv2.waitKey(25) & 0xFF == ord('q'):
          break
 
cap.release()
cv2.destroyAllWindows()
