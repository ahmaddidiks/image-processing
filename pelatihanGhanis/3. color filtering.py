# memasukkan library yang akan digunakan
import cv2
import numpy as np


def nothing(x):
    pass


# mulai video capture
cap = cv2.VideoCapture(1);

# setting resolusi video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# buat windows
cv2.namedWindow('BGR')
cv2.namedWindow('Hue')
cv2.namedWindow('Saturation')
cv2.namedWindow('Value')
cv2.namedWindow('Hasil')

# buat trackbar untuk atur nilai threshold
cv2.createTrackbar('hmin', 'Hue', 0, 179, nothing)
cv2.createTrackbar('hmax', 'Hue', 179, 179, nothing)
cv2.createTrackbar('smin', 'Saturation', 0, 255, nothing)
cv2.createTrackbar('smax', 'Saturation', 255, 255, nothing)
cv2.createTrackbar('vmin', 'Value', 0, 255, nothing)
cv2.createTrackbar('vmax', 'Value', 255, 255, nothing)

# Ambil gambar terus menerus sampai selesai
while True:
    # ambil gambar dari kamera
    ret, bgr_image = cap.read()

    # ubah ke hsv
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # split gambar ke masing-masing channel
    hue, sat, val = cv2.split(hsv_image)

    # ambil nilai threshold dari trackbar
    hmin = cv2.getTrackbarPos('hmin', 'Hue')
    hmax = cv2.getTrackbarPos('hmax', 'Hue')
    smin = cv2.getTrackbarPos('smin', 'Saturation')
    smax = cv2.getTrackbarPos('smax', 'Saturation')
    vmin = cv2.getTrackbarPos('vmin', 'Value')
    vmax = cv2.getTrackbarPos('vmax', 'Value')

    # filter warna menggunakan thresholding
    hThresh = cv2.inRange(np.array(hue), np.array(hmin), np.array(hmax))
    sThresh = cv2.inRange(np.array(sat), np.array(smin), np.array(smax))
    vThresh = cv2.inRange(np.array(val), np.array(vmin), np.array(vmax))

    # gabungkan ketiga channel yang sudah difilter
    hsvThresh = cv2.bitwise_and(hThresh, cv2.bitwise_and(sThresh, vThresh))
 
    # tampilkan gambar
    cv2.imshow('BGR', bgr_image)
    cv2.imshow('Hue', hThresh)
    cv2.imshow('Saturation', sThresh)
    cv2.imshow('Value', vThresh)
    cv2.imshow('Hasil', hsvThresh)
 
    # apabila tombol q ditekan maka perulangan berhenti
    if cv2.waitKey(25) & 0xFF == ord('q'):
          break
 
# selesai ambil gambar
cap.release()
 
# tutup semua windows
cv2.destroyAllWindows()

