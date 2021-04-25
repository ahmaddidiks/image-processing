# memasukkan library yang akan digunakan
import cv2
import numpy as np

# load gambar
bgr_image = cv2.imread('./gambar/undip.jpg')

# ubah ke hsv
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# tampilkan gambar
cv2.imshow('Model BGR', bgr_image)
cv2.imshow('Model HSV', hsv_image)

# tutup window apabila keyboard ditekan
cv2.waitKey()
cv2.destroyAllWindows()
