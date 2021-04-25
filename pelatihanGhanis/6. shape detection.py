import cv2
import numpy as np


def nothing(x):
    pass


# fungsi deteksi bentuk
# input: c = contour, const = parameter untuk approxPolyDP
# output: bentuk benda
def detect(c, const):
    shape = "unidentified"
    peri = cv2.arcLength(c, True) #keliling true=kontur terbuka, false=konture terbuka
    approx = cv2.approxPolyDP(c, const * peri, True) #

    #aprox=menyimpan berapa sudut benda, segi 4 => ada 4 sudut
    if len(approx) == 3:
        shape = "segitiga"

    elif len(approx) == 4:
        shape = "segiempat"

    elif len(approx) == 5:
        shape = "segilima"

    else:
        shape = "lingkaran"

    return shape


kernel1 = np.ones((1, 1), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(0);

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cv2.namedWindow('BGR')
cv2.namedWindow('Hue')
cv2.namedWindow('Saturation')
cv2.namedWindow('Value')
cv2.namedWindow('Filtering')

cv2.createTrackbar('hmin', 'Hue', 0, 179, nothing)
cv2.createTrackbar('hmax', 'Hue', 179, 179, nothing)
cv2.createTrackbar('smin', 'Saturation', 0, 255, nothing)
cv2.createTrackbar('smax', 'Saturation', 255, 255, nothing)
cv2.createTrackbar('vmin', 'Value', 0, 255, nothing)
cv2.createTrackbar('vmax', 'Value', 255, 255, nothing)
cv2.createTrackbar('konstanta', 'BGR', 10, 100, nothing)

while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    hmin = cv2.getTrackbarPos('hmin', 'Hue')
    hmax = cv2.getTrackbarPos('hmax', 'Hue')
    smin = cv2.getTrackbarPos('smin', 'Saturation')
    smax = cv2.getTrackbarPos('smax', 'Saturation')
    vmin = cv2.getTrackbarPos('vmin', 'Value')
    vmax = cv2.getTrackbarPos('vmax', 'Value')
    const = cv2.getTrackbarPos('konstanta', 'BGR')

    per_const = const/100
    
    hThresh = cv2.inRange(np.array(hue), np.array(hmin), np.array(hmax))
    sThresh = cv2.inRange(np.array(sat), np.array(smin), np.array(smax))
    vThresh = cv2.inRange(np.array(val), np.array(vmin), np.array(vmax))

    hsvThresh = cv2.bitwise_and(hThresh, cv2.bitwise_and(sThresh, vThresh))
    
    hsvThresh = cv2.morphologyEx(hsvThresh, cv2.MORPH_OPEN, kernel1)
    hsvThresh = cv2.morphologyEx(hsvThresh, cv2.MORPH_DILATE, kernel2)

    edges = cv2.Canny(hsvThresh, 0, 0)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        for c in contours:  #ada banyak kontur, maka digunakan perulangan
            M = cv2.moments(c)
            if M["m00"] != 0:                       #
                cX = int((M["m10"] / M["m00"]))     #}center of mass(titik pusat gambar)
                cY = int((M["m01"] / M["m00"]))     #}
            else:
                cX, cY = 0, 0

            #deteksi bentuk
            text = detect(c, per_const)

            c = c.astype("int")
            cv2.drawContours(frame, [c], -1, (255, 0, 0), 3)
            cv2.putText(frame, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #putText

    cv2.imshow('BGR', frame)
    cv2.imshow('Filtering', hsvThresh)
    cv2.imshow('Hue', hThresh)
    cv2.imshow('Saturation', sThresh)
    cv2.imshow('Value', vThresh)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
