import cv2
import numpy as np


def nothing(x):
    pass


# fungsi untuk menghitung similarity
# similarity = luas kontur / luas lingkaran
# input: cnt = kontur, radius = jari-jari dari kontur
# output: nilai similarity
def Similarity(cnt, radius):
    ContourArea = cv2.contourArea(cnt)
    CircleArea = float(np.pi * radius ** 2)
    if CircleArea > 0:
        value = float(ContourArea / CircleArea)
    else:
        value = 0

    return value


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cv2.namedWindow('Hasil')
cv2.namedWindow('HSV')
cv2.namedWindow('Hue')
cv2.namedWindow('Saturation')
cv2.namedWindow('Value')
cv2.namedWindow('Edge Detector')

cv2.createTrackbar('hmin', 'Hue', 0, 179, nothing)
cv2.createTrackbar('hmax', 'Hue', 179, 179, nothing)
cv2.createTrackbar('smin', 'Saturation', 0, 255, nothing)
cv2.createTrackbar('smax', 'Saturation', 255, 255, nothing)
cv2.createTrackbar('vmin', 'Value', 0, 255, nothing)
cv2.createTrackbar('vmax', 'Value', 255, 255, nothing)
cv2.createTrackbar('Min Radius', 'Hasil', 0, 100, nothing)
cv2.createTrackbar('Max Radius', 'Hasil', 100, 100, nothing)
cv2.createTrackbar('Similarity', 'Hasil', 0, 100, nothing)

while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    hmn = cv2.getTrackbarPos('hmin', 'Hue')
    hmx = cv2.getTrackbarPos('hmax', 'Hue')
    smn = cv2.getTrackbarPos('smin', 'Saturation')
    smx = cv2.getTrackbarPos('smax', 'Saturation')
    vmn = cv2.getTrackbarPos('vmin', 'Value')
    vmx = cv2.getTrackbarPos('vmax', 'Value')
    sim_thresh = cv2.getTrackbarPos('Similarity', 'Hasil')
    minRadius = cv2.getTrackbarPos('Min Radius', 'Hasil')
    maxRadius = cv2.getTrackbarPos('Max Radius', 'Hasil')

    per_sim_thresh = float(sim_thresh)/100

    hthresh = cv2.inRange(np.array(hue), np.array(hmn), np.array(hmx))
    sthresh = cv2.inRange(np.array(sat), np.array(smn), np.array(smx))
    vthresh = cv2.inRange(np.array(val), np.array(vmn), np.array(vmx))

    hsvThresh = cv2.bitwise_and(hthresh, cv2.bitwise_and(sthresh, vthresh))

    edges = cv2.Canny(hsvThresh, 0, 0)

    _, contours, _= cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] #kontur ada 5
    if len(contours) > 0:
        intRadius = 0
        center = (0, 0)
        similarity = 0
        for i in range(len(contours)):
            cnt = contours[i]

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            intRadius = int(radius)
            similarity = Similarity(cnt, radius)

            cv2.circle(frame, center, intRadius, (0, 0, 255), 1)
            if (minRadius < intRadius < maxRadius) and (similarity > per_sim_thresh):
                xText = 'X: ' + str(center[0])
                yText = 'Y: ' + str(center[1])
                cv2.putText(frame, xText, (2, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, yText, (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.circle(frame, center, intRadius, (255, 0, 0), 2) #untuk menggambar didalam frame
                cv2.putText(frame, "Terdeteksi", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break

    cv2.imshow('Hasil', frame)
    cv2.imshow('HSV', hsvThresh)
    cv2.imshow('Edge Detector', edges)
    cv2.imshow('Hue', hthresh)
    cv2.imshow('Saturation', sthresh)
    cv2.imshow('Value', vthresh)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
