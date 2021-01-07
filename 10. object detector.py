import cv2
import numpy as np

MIN_MATCH_COUNT=18

detector = cv2.ORB_create( nfeatures = 1000 )

bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING)

trainImg = cv2.imread("gambar/didik.jpg",0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(gray, None)

    matches=bfMatcher.knnMatch(queryDesc, trainDesc, k=2)
    goodMatch = []
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)

    cv2.putText(frame, str(len(goodMatch)), (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

    if(len(goodMatch) > MIN_MATCH_COUNT):
        tp=[]
        qp=[]

        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)

        tp, qp = np.float32((tp,qp))
        H, _ = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0) #menghtung vektor tranformasi, mendapatkan benda bisadilihat dari amping
        h, w = trainImg.shape
        trainBorder = np.float32([[[0,0], [0,h-1], [w-1,h-1], [w-1,0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(frame, [np.int32(queryBorder)], True, (255,0,0), 2) #membuatgaris dari titik pojok/sudut

    cv2.imshow('Hasil',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
