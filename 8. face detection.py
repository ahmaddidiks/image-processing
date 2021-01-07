
import cv2
import numpy as np

# detector wajah dan mata
deteksi_wajah = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
deteksi_mata = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # deteksi wajah
    wajah = deteksi_wajah.detectMultiScale(gray, 1.1, 1)
    banyak= "bnayak wajah terdeteksi : " + str(len(wajah))
    cv2.putText(frame, banyak, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1 )


    for (x,y,w,h) in wajah:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # deteksi mata
        mata = deteksi_mata.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in mata:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            
    cv2.imshow('Hasil',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

