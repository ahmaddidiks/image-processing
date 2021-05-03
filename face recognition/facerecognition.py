from time import time
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import cv2 as cv
import sys
import matplotlib.pyplot as plt

path = "/home/didik/image-processing/face recognition/dataset/"

data_slice = [70, 195, 78, 172] #[ymin, ymax, xmin, xmax]

#to sxtrac the 'instersting' part of the image files and avoid use statistical correlation from the background

#resize the ratio to reduce sample dimention
resize_ratio = 2.5

height = int((data_slice[1] - data_slice[0])/resize_ratio) #ymax - ymin slice, Height of th the image in float
width = int((data_slice[3] - data_slice[2])/resize_ratio) #xmax - xmin, width of the image in float
print(f"Image dimention after resize (h,w) : {height,width}")
n_sample = 0 #initial sample count
label_count = 0 #intial label count
n_classes = 0 #initial class count

#PCA COmponent
n_components = 7 #7

#Flat image feature vector
x=[]
#int array of label vector
y=[]

target_names = []#Array to store the name of people

for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        img = cv.imread(path+directory+"/"+file)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
        img = cv.resize(img, (width,height))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        featureVector= np.array(img).flatten()
        x.append(featureVector)
        y.append(label_count)
        n_sample +=1
    
    target_names.append(directory)
    label_count +=1

print(f"Samples : {n_sample}")
print(f"Class   : {target_names}")
n_classes = len(target_names)

#Split into a traini gset and a test set using startified k fold

#split into a training and testing set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)

#compute a PCA (eigenfaces) on the face dataset (treated as unlabeled dataset): unsupervised feature extraction/dimensionality reduction

print(f"Extracting the top {n_components} eaigenfaces from {len(x_train)} faces")

t0= time()
pca = PCA(n_components=n_components, whiten=True).fit(x_train)
print(f"Done in {time() - t0} s")

eigenfaces = pca.components_.reshape((n_components, height, width))
print("\nProjecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print(f"Done in {time()-t0}")

def plot_gallery(images, titles, height, width, n_row=3, n_col=4):
    """Helper function to plot a gallery of potraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.1, right=.99, top=.90, hspace=.35)
    for i in range(n_components):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigen_face_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigen_face_titles, height, width)

#Train a SVM classification model
print("\nFitting the classifier to the training set")
t0=time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [.0001, .0005, .001, .005, .01, .01],}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),param_grid, verbose=5)
clf = clf.fit(x_train_pca, y_train)
print(f"Done in {time()-t0}s")

print(f"\nBest estiamtor found by grid search : {clf.best_estimator_}")

#Quantitive evaluation of the model quality on the test set
print("\nPredicting people's names on the test set")
t0=time()
y_predict=clf.predict(x_test_pca)
print(clf.score(x_test_pca, y_test))
print(f"Done in {time()-t0}s")

print("\nClassification Report : ")
print(classification_report(y_test, y_predict, target_names=target_names))

print("\nConfusion Matrix : ")
print(confusion_matrix(y_test, y_predict, labels=range(n_classes)))

#Prediction of user based on the model

test = []
test_image = "/home/didik/image-processing/face recognition/dataset/Tiger_Woods/Tiger_Woods_0002.jpg"
test_image = cv.imread(test_image)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
test_image = cv.resize(test_image, (width,height))
test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
test_image_feature_vector = np.array(test_image).flatten()
test.append(test_image_feature_vector)
test_image_pca= pca.transform(test)
test_image_predict = clf.predict(test_image_pca)

print(f"Prdicted Name : {target_names[test_image_predict[0]]}")

path_data_set = "test/"
#path_data_set = "/home/didik/image-processing/face recognition/dataset/"
for file in os.listdir(path_data_set):
    test = []
    file_name = path_data_set + file
    test_image = cv.imread(file_name)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
    test_image = cv.resize(test_image, (width,height))
    test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    test_image_feature_vector = np.array(test_image).flatten()

    test.append(test_image_feature_vector)
    test_image_pca = pca.transform(test)
    test_image_predict = clf.predict(test_image_pca)
    print(f"File Source : {file_name}")
    print(f"Predicted Name : {target_names[test_image_predict[0]]}\n")

face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
while(True):
    #capture fram by frame
    test = []
    face = []
    ret, frame = cap.read()
    print(type(frame))
    xv, yv, vc = frame.shape
    if (ret):
        
       # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for(x,y,wf,hf) in faces:
            cy, cx = y + (hf//2), x+(wf//2)
            max_len = max(max(hf//2, wf//2), 125)
            
            if(x-max_len) <= 0 or (x+max_len) >= xv or (y-max_len) <= 0 or (y+max_len) >= yv:
                continue
            
            face_crop = (frame[cy-max_len:cy+max_len, cx-max_len:cx+max_len])
            face_crop = face_crop[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]

            test_image = cv.resize(face_crop, (width, height))
            cv.imshow('face', test_image)
            test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
            test_image_feature_vector = np.array(test_image).flatten()

            test.append(test_image_feature_vector)
            test_image_pca = pca.transform(test)
            test_image_predict = clf.predict(test_image_pca)

            #create box on detected face
            frame = cv.rectangle(frame, (x,y), (x+wf, y+hf), (255,0,0),1)
            frame = cv.rectangle(frame,(x,y+hf), (x+wf, y+hf+30), (255,0,0),-1)

            #print label name on image
            cv.putText(frame, "Name : " + target_names[test_image_predict[0]], (x + x//10, y+hf+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

        cv.imshow('frame',frame)

        if cv.waitkey(1) & 0xFF == ord('q'):
            break

cap.realease()
cv.destroyAllWindows()