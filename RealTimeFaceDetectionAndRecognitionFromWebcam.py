import numpy as np
import cv2
from random import *
import pandas as pd
from PIL import Image
from sklearn import svm
import os

def processImage(filename):
    im=Image.open(filename)
    im=im.convert('1')
    im = im.resize((128,128), Image.ANTIALIAS)
    data=list(im.getdata())
    return data

def processImages(foldername,classifierindex):
    for file in os.listdir(foldername):
        filename=foldername+file
        data=processImage(filename)
        X.append(data)
        y.append(classifierindex)
        print("Processed file "+filename)

X=[]
y=[]
outputs=["mita","bisho"]


def testCroppedData(crop_img):
    cv2.imwrite('tmp.png',crop_img)
    im2=Image.open('tmp.png')
    im2=im2.convert('1')
    im2 = im2.resize((128,128), Image.ANTIALIAS)
    data=list(im2.getdata())
    y_pred=clf.predict([data])
    os.remove('tmp.png')
    print(y_pred)
    return outputs[y_pred[0]]


processImages("bisho/",1)
processImages("mita/",0)



clf=svm.SVC(kernel="linear",gamma=1)
clf.fit(X,y)



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(1)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_found=0
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(5,255,40),12)
        roi_color = img[y:y+h, x:x+w]
        crop_img = img[y:y+h, x:x+w]
        cv2.putText(img,testCroppedData(crop_img), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(128,128,0),4)
        face_found=1
        #crop_img = img[y:y+h, x:x+w]
        #cv2.imshow(str(randint(0,1000))+"cropped", crop_img)

    img = cv2.resize(img, (840, 680)) 
    cv2.imshow('img',img)
    k=cv2.waitKey(60) & 0xff
    if k == 27:
        break
    
    

cap.release()
cv2.destroyAllWindows()
