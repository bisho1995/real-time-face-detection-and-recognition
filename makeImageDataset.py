import numpy as np
import cv2
from random import *

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('068.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
face_found=0
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(5,255,40),8)
    roi_color = img[y:y+h, x:x+w]
    cv2.putText(img,"Face", (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.8,(128,128,0),2)
    face_found=1
    #crop_img = img[y:y+h, x:x+w]
    #cv2.imshow(str(randint(0,1000))+"cropped", crop_img)

if face_found==1:
    print("face found")
    img = cv2.resize(img, (840, 680)) 
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected")
