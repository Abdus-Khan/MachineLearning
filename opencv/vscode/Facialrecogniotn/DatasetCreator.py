import cv2
import numpy as np  

faces_cascade = cv2.CascadeClassifier('/Users/abdus/Desktop/machine_learning/opencv/vscode/Facialrecogniotn/haarcascade_frontalface_default.xml')

id = input ('input your id')
sampleNum = 0
cam = cv2.VideoCapture (0)
while True:
    ret , frame = cam.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale (gray , 1.3 , 5)

    for (x ,y ,w ,h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w ,y+h) , (255 ,0,0) , 2)
        sampleNum = sampleNum+1
        cv2.imwrite ("/Users/abdus/Desktop/machine_learning/opencv/vscode/Facialrecogniotn/dataSet" +str(id)+'.'+ str(sampleNum) +'.jpg'  , gray[y:y+h , x:x+h] )
        cv2.imshow ('video' , frame)

    if cv2.waitKey(500) & 0xff == ord('q'):
        break
    elif sampleNum>200:
        break
    cam.release
    cv2.destroyAllWindows    