import cv2 as cv
import numpy as np

image = cv.imread('gambar/didik.jpg')

#ini konversinya
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
gray_image= cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#ini menampilkan gambar
cv.imshow('Model BGR', image)
cv.imshow('Model HSV', hsv_image)
cv.imshow('Model GRAYSCALE', gray_image)

cv.waitKey()
cv.destroyAllWindows()

class gambar:
    def __init__(self, nama):
        gambar.nama  = nama
        