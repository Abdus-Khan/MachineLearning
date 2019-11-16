import cv2
import numpy as np 
import os
from PIL import Image
import glob

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = '/Users/abdus/Desktop/machine_learning/opencv/vscode/Facialrecogniotn/dataSet'  

def getImagesWithIds(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    if path + '.DS_Store' in imagePaths:
     imagePaths.remove(path + '.DS_Store')
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceimg =  Image.open(imagePath).convert('L')
        faceNp = np.array(faceimg , 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('images' , faceNp)
        cv2.waitKey(10)
    return np.array(IDs) , faces
ids ,faces = getImagesWithIds(path)
recognizer.train(faces,ids)
recognizer.save('/Users/abdus/Desktop/machine_learning/opencv/vscode/Facialrecogniotn/recognizer/training.yml')
cv2.destroyAllWindows()

