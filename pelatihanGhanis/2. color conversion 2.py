# memasukkan library yang akan digunakan
import cv2

# mulai video capture
cap = cv2.VideoCapture(0);

# setting resolusi video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# ID number ten for brightness
cap.set(10, 70)

# Ambil gambar terus menerus sampai selesai
while True:
    # ambil gambar dari kamera
    ret, bgr_image = cap.read()

    # ubah ke hsv
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
 
    # tampilkan gambar
    cv2.imshow('BGR', bgr_image)
    cv2.imshow('HSV', hsv_image)
 
    # apabila tombol q ditekan maka perulangan berhenti
    if cv2.waitKey(25) & 0xFF == ord('q'):
          break
 
# selesai ambil gambar
cap.release()
 
# tutup semua windows
cv2.destroyAllWindows()

