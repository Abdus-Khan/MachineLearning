import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('/Users/abdus/Desktop/machine_learning/opencv/vscode/Facialrecogniotn/haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("/Users/abdus/Desktop/machine_learning/opencv/vscode/Facialrecogniotn/recognizer/training.yml")
id1 = [1,2,3,4,5]
id2 = [6,7,8,9,10]
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
gender = ' Gender = Male'
offset = -50
ofset3 = -100
offfset4 = -150
age = ' age = 8'
Crimimalrecord = 'Criminal Record = making illegal robots'
while(True):
    ret,img=cam.read()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (id == 1 ,2 ):
            id = "Name:Abdus"
            age = 15
        elif(id == 3 ,4):
            id = 'Name : Zaid'
        else:
            id == 'Uknown'
        cv2.putText(img, str(id), (x, y + h), fontface, fontscale, fontcolor)
        cv2.putText(img, str(gender),(x , y + h - offset), fontface, fontscale, fontcolor)
        cv2.putText(img, str(age),(x , y + h - ofset3), fontface, fontscale, fontcolor)
        cv2.putText(img, str(Crimimalrecord),(x , y + h - offfset4), fontface, fontscale, fontcolor)
    cv2.imshow("Face", img)
    if (cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
                                                                          #
                                                                          #